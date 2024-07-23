using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using TorchSharp;
using static TorchSharp.torch;
using RouteLLM.Core;

namespace RouteLLM.Routers
{
    [AttributeUsage(AttributeTargets.Class)]
    public class NoParallelAttribute : Attribute { }

    public abstract class Router : IRouter
    {
        public virtual bool NoParallel => GetType().GetCustomAttributes(typeof(NoParallelAttribute), true).Length > 0;

        public abstract Task<float> CalculateStrongWinRate(string prompt);

        public virtual async Task<string> Route(string prompt, float threshold, ModelPair routedPair)
        {
            float winRate = await CalculateStrongWinRate(prompt);
            return winRate >= threshold ? routedPair.Strong : routedPair.Weak;
        }

        public override string ToString()
        {
            return GetType().Name;
        }
    }

    [NoParallel]
    public class RandomRouter : Router
    {
        private readonly Random _random = new Random();

        public override Task<float> CalculateStrongWinRate(string prompt)
        {
            return Task.FromResult((float)_random.NextDouble());
        }
    }

    [NoParallel]
    public class CausalLLMRouter : Router
    {
        private readonly CausalLLMClassifier classifier;
        private readonly string systemMessage;
        private readonly string classifierMessage;

        public CausalLLMRouter(
            string checkpointPath,
            float scoreThreshold = 4,
            string[] specialTokens = null,
            int numOutputs = 5,
            string modelType = "causal",
            string modelId = "meta-llama/Meta-Llama-3-8B",
            bool flashAttention2 = false)
        {
            specialTokens ??= new[] { "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" };
            this.classifier = new CausalLLMClassifier(checkpointPath, specialTokens, scoreThreshold);

            // Load system and classifier messages
            this.systemMessage = File.ReadAllText(Path.Combine(checkpointPath, "system_ft_v5.txt"));
            this.classifierMessage = File.ReadAllText(Path.Combine(checkpointPath, "classifier_ft_v5.txt"));
        }

        public override async Task<float> CalculateStrongWinRate(string prompt)
        {
            var formattedPrompt = FormatPrompt(prompt);
            var binaryProb = classifier.Predict(formattedPrompt);

            if (binaryProb == null)
            {
                // Route to strong model if output is invalid
                return 1;
            }
            else
            {
                return 1 - binaryProb.Value;
            }
        }

        private string FormatPrompt(string prompt)
        {
            var messages = new List<Dictionary<string, string>>
            {
                new Dictionary<string, string> { { "role", "system" }, { "content", systemMessage } },
                new Dictionary<string, string> { { "role", "user" }, { "content", prompt } },
                new Dictionary<string, string> { { "role", "assistant" }, { "content", classifierMessage } }
            };

            return JsonSerializer.Serialize(messages);
        }
    }

    [NoParallel]
    public class BERTRouter : Router
    {
        private readonly MLContext mlContext;
        private readonly PredictionEngine<BERTInputData, BERTOutputData> predictionEngine;

        public BERTRouter(string checkpointPath, int numLabels = 3)
        {
            this.mlContext = new MLContext();

            // Load the model
            ITransformer mlModel = mlContext.Model.Load(checkpointPath, out var _);
            this.predictionEngine = mlContext.Model.CreatePredictionEngine<BERTInputData, BERTOutputData>(mlModel);
        }

        public override async Task<float> CalculateStrongWinRate(string prompt)
        {
            var input = new BERTInputData { Text = prompt };
            var output = predictionEngine.Predict(input);

            // Compute prob of label 1 and 2 (tie, tier 2 wins)
            float binaryProb = output.Probabilities[1] + output.Probabilities[2];
            return 1 - binaryProb;
        }

        private class BERTInputData
        {
            [VectorType(1)]
            public string Text { get; set; }
        }

        private class BERTOutputData
        {
            [VectorType(3)]
            public float[] Probabilities { get; set; }
        }
    }

    public class SWRankingRouter : Router
    {
        private readonly Dictionary<string, int> model2tier;
        private readonly string strongModel;
        private readonly string weakModel;
        private readonly float[,] arenaConvEmbedding;
        private readonly string embeddingModel = "text-embedding-3-small";
        private readonly List<ArenaDataItem> arenaDF;

        public SWRankingRouter(
            List<string> arenaBattleDatasets,
            List<string> arenaEmbeddingDatasets,
            string strongModel = "gpt-4-1106-preview",
            string weakModel = "mixtral-8x7b-instruct-v0.1",
            int numTiers = 10)
        {
            this.strongModel = strongModel;
            this.weakModel = weakModel;

            this.arenaDF = SWRankingUtils.LoadAndPreprocessArenaDatasets(arenaBattleDatasets);
            this.arenaConvEmbedding = SWRankingUtils.LoadArenaEmbeddings(arenaEmbeddingDatasets);

            var modelRatings = SWRankingUtils.ComputeEloMleWithTie(this.arenaDF);
            this.model2tier = SWRankingUtils.ComputeTiers(modelRatings, numTiers);

            SWRankingUtils.UpdateArenaDataFrameWithTiers(this.arenaDF, this.model2tier);
        }

        public override async Task<float> CalculateStrongWinRate(string prompt)
        {
            var promptEmb = await SWRankingUtils.GetEmbedding(prompt, embeddingModel);
            var similarities = SWRankingUtils.ComputeSimilarities(promptEmb, arenaConvEmbedding);
            var weightings = GetWeightings(similarities);

            var res = SWRankingUtils.ComputeEloMleWithTie(arenaDF, weightings);

            float weakScore = res[model2tier[weakModel]];
            float strongScore = res[model2tier[strongModel]];
            float weakWinrate = 1 / (1 + (float)Math.Pow(10, (strongScore - weakScore) / 400));
            float strongWinrate = 1 - weakWinrate;

            return strongWinrate;
        }

        private float[] GetWeightings(float[] similarities)
        {
            float maxSim = similarities.Max();
            return similarities.Select(s => (float)(10 * Math.Pow(10, s / maxSim))).ToArray();
        }
    }

    [NoParallel]
    public class MatrixFactorizationRouter : Router
    {
        private readonly Module model;
        private readonly int strongModelId;
        private readonly int weakModelId;
        private readonly string checkpointPath;
        private readonly int hiddenSize;
        private readonly int numModels;
        private readonly int textDim;
        private readonly int numClasses;
        private readonly bool useProj;

        private static readonly Dictionary<string, int> MODEL_IDS = new Dictionary<string, int>
        {
            {"gpt-4-1106-preview", 0},
            {"mixtral-8x7b-instruct-v0.1", 1},
            // Add other model IDs here
        };

        public MatrixFactorizationRouter(
            string checkpointPath,
            string strongModel = "gpt-4-1106-preview",
            string weakModel = "mixtral-8x7b-instruct-v0.1",
            int hiddenSize = 128,
            int numModels = 64,
            int textDim = 1536,
            int numClasses = 1,
            bool useProj = true)
        {
            this.checkpointPath = checkpointPath;
            this.hiddenSize = hiddenSize;
            this.numModels = numModels;
            this.textDim = textDim;
            this.numClasses = numClasses;
            this.useProj = useProj;

            // Load the model using TorchSharp
            this.model = torch.jit.load(checkpointPath);

            if (!MODEL_IDS.TryGetValue(strongModel, out this.strongModelId))
            {
                throw new ArgumentException($"Unknown strong model: {strongModel}");
            }

            if (!MODEL_IDS.TryGetValue(weakModel, out this.weakModelId))
            {
                throw new ArgumentException($"Unknown weak model: {weakModel}");
            }
        }

        public override async Task<float> CalculateStrongWinRate(string prompt)
        {
            using (torch.no_grad())
            {
                var promptTensor = await GetPromptEmbedding(prompt);
                var inputDict = new Dictionary<string, Tensor>
                {
                    {"prompt_emb", promptTensor},
                    {"model_a", torch.tensor(strongModelId)},
                    {"model_b", torch.tensor(weakModelId)}
                };

                var output = model.forward(inputDict);
                return output.item<float>();
            }
        }

        private async Task<Tensor> GetPromptEmbedding(string prompt)
        {
            var embedding = await SWRankingUtils.GetEmbedding(prompt, "text-embedding-3-small");
            return torch.tensor(embedding);
        }

        public async Task Train(List<TrainingDataItem> trainingData, int batchSize = 64, int numEpochs = 100, float learningRate = 3e-4, float weightDecay = 1e-5)
        {
            var dataset = new PairwiseDataset(trainingData);
            var dataLoader = dataset.GetDataLoader(batchSize, shuffle: true);

            var optimizer = torch.optim.Adam(model.parameters(), lr: learningRate, weight_decay: weightDecay);
            var lossFunction = torch.nn.BCEWithLogitsLoss();

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                float totalLoss = 0;
                int batchCount = 0;

                foreach (var batch in dataLoader)
                {
                    optimizer.zero_grad();

                    var promptEmb = batch["prompt_emb"];
                    var modelA = batch["model_a"];
                    var modelB = batch["model_b"];
                    var labels = batch["label"];

                    var output = model.forward(new Dictionary<string, Tensor>
                    {
                        {"prompt_emb", promptEmb},
                        {"model_a", modelA},
                        {"model_b", modelB}
                    });

                    var loss = lossFunction.forward(output, labels);
                    loss.backward();
                    optimizer.step();

                    totalLoss += loss.item<float>();
                    batchCount++;
                }

                Console.WriteLine($"Epoch {epoch + 1}/{numEpochs}, Average Loss: {totalLoss / batchCount}");
            }

            // Save the trained model
            model.save(checkpointPath);
        }
    }

    public class TrainingDataItem
    {
        public float[] PromptEmbedding { get; set; }
        public int ModelA { get; set; }
        public int ModelB { get; set; }
        public float Label { get; set; }
    }

    public class PairwiseDataset
    {
        private readonly List<TrainingDataItem> data;

        public PairwiseDataset(List<TrainingDataItem> data)
        {
            this.data = data;
        }

        public IEnumerable<Dictionary<string, Tensor>> GetDataLoader(int batchSize, bool shuffle)
        {
            var indices = Enumerable.Range(0, data.Count).ToList();
            if (shuffle)
            {
                indices = indices.OrderBy(x => Guid.NewGuid()).ToList();
            }

            for (int i = 0; i < data.Count; i += batchSize)
            {
                var batchIndices = indices.Skip(i).Take(batchSize).ToList();
                var batchData = batchIndices.Select(idx => data[idx]).ToList();

                yield return new Dictionary<string, Tensor>
                {
                    {"prompt_emb", torch.tensor(batchData.Select(d => d.PromptEmbedding).ToArray())},
                    {"model_a", torch.tensor(batchData.Select(d => d.ModelA).ToArray())},
                    {"model_b", torch.tensor(batchData.Select(d => d.ModelB).ToArray())},
                    {"label", torch.tensor(batchData.Select(d => d.Label).ToArray())}
                };
            }
        }
    }
}