using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Net.Http;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using System.IO;
using RouteLLM.Core;

namespace RouteLLM.Routers
{
    public static class SWRankingUtils
    {
        public static List<ArenaDataItem> LoadAndPreprocessArenaDatasets(List<string> arenaBattleDatasets)
        {
            var allData = new List<ArenaDataItem>();

            foreach (var dataset in arenaBattleDatasets)
            {
                var jsonData = File.ReadAllText(dataset);
                var dataItems = JsonSerializer.Deserialize<List<ArenaDataItem>>(jsonData);
                allData.AddRange(dataItems);
            }

            return allData.Where(item => item.Winner != "tie" && item.ModelA != item.ModelB).ToList();
        }

        public static float[,] LoadArenaEmbeddings(List<string> arenaEmbeddingDatasets)
        {
            var allEmbeddings = new List<float[]>();

            foreach (var dataset in arenaEmbeddingDatasets)
            {
                var embeddings = File.ReadAllLines(dataset)
                    .Select(line => line.Split(',').Select(float.Parse).ToArray())
                    .ToList();
                allEmbeddings.AddRange(embeddings);
            }

            var result = new float[allEmbeddings.Count, allEmbeddings[0].Length];
            for (int i = 0; i < allEmbeddings.Count; i++)
            {
                for (int j = 0; j < allEmbeddings[i].Length; j++)
                {
                    result[i, j] = allEmbeddings[i][j];
                }
            }

            return result;
        }

        public static Dictionary<string, float> ComputeEloMleWithTie(List<ArenaDataItem> arenaData, float[] weightings = null)
        {
            var uniqueModels = arenaData.SelectMany(x => new[] { x.ModelA, x.ModelB }).Distinct().ToList();
            var modelToIndex = uniqueModels.ToDictionary(x => x, x => uniqueModels.IndexOf(x));

            var n = uniqueModels.Count;
            var initialGuess = CreateVector.Dense(n, 1500.0);

            var objective = ObjectiveFunction.Value(ratings =>
            {
                double logLikelihood = 0.0;
                for (int i = 0; i < arenaData.Count; i++)
                {
                    var battle = arenaData[i];
                    int modelAIndex = modelToIndex[battle.ModelA];
                    int modelBIndex = modelToIndex[battle.ModelB];
                    double ratingDiff = ratings[modelAIndex] - ratings[modelBIndex];
                    double prob = 1.0 / (1.0 + Math.Exp(-ratingDiff / 400.0));

                    double weight = weightings != null ? weightings[i] : 1.0;

                    if (battle.Winner == battle.ModelA)
                        logLikelihood += weight * Math.Log(prob);
                    else
                        logLikelihood += weight * Math.Log(1 - prob);
                }
                return -logLikelihood;
            });

            var solver = new BfgsBMinimizer(1e-5, 1000);
            var result = solver.FindMinimum(objective, initialGuess);

            return uniqueModels.ToDictionary(model => model, model => (float)result.MinimizingPoint[modelToIndex[model]]);
        }

        public static Dictionary<string, int> ComputeTiers(Dictionary<string, float> modelRatings, int numTiers)
        {
            var sortedRatings = modelRatings.OrderByDescending(x => x.Value).ToList();
            var tierSize = (int)Math.Ceiling((double)sortedRatings.Count / numTiers);

            var model2tier = new Dictionary<string, int>();
            for (int i = 0; i < sortedRatings.Count; i++)
            {
                model2tier[sortedRatings[i].Key] = i / tierSize;
            }

            return model2tier;
        }

        public static void UpdateArenaDataFrameWithTiers(List<ArenaDataItem> arenaDF, Dictionary<string, int> model2tier)
        {
            foreach (var item in arenaDF)
            {
                item.ModelATier = model2tier[item.ModelA];
                item.ModelBTier = model2tier[item.ModelB];
            }
        }

        public static async Task<float[]> GetEmbedding(string prompt, string embeddingModel)
        {
            return await EmbeddingUtils.GetEmbedding(prompt, embeddingModel);
        }

        public static float[] ComputeSimilarities(float[] promptEmb, float[,] arenaConvEmbedding)
        {
            int numEmbeddings = arenaConvEmbedding.GetLength(0);
            float[] similarities = new float[numEmbeddings];

            for (int i = 0; i < numEmbeddings; i++)
            {
                float[] embeddingVector = new float[arenaConvEmbedding.GetLength(1)];
                for (int j = 0; j < embeddingVector.Length; j++)
                {
                    embeddingVector[j] = arenaConvEmbedding[i, j];
                }
                similarities[i] = EmbeddingUtils.CosineSimilarity(promptEmb, embeddingVector);
            }

            return similarities;
        }

        private static List<ArenaDataItem> LoadDataset(string dataset)
        {
            var jsonString = File.ReadAllText(dataset);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            return JsonSerializer.Deserialize<List<ArenaDataItem>>(jsonString, options);
        }

        private static List<float[]> LoadEmbeddings(string dataset)
        {
            var jsonString = File.ReadAllText(dataset);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            return JsonSerializer.Deserialize<List<float[]>>(jsonString, options);
        }

        private static float[,] ConvertToMatrix(List<float[]> embeddings)
        {
            var result = new float[embeddings.Count, embeddings[0].Length];
            for (int i = 0; i < embeddings.Count; i++)
            {
                for (int j = 0; j < embeddings[i].Length; j++)
                {
                    result[i, j] = embeddings[i][j];
                }
            }
            return result;
        }

        private static float[] GetRow(float[,] matrix, int row)
        {
            return Enumerable.Range(0, matrix.GetLength(1))
                .Select(x => matrix[row, x])
                .ToArray();
        }

        private static List<ArenaDataItem> PreprocessBattles(List<ArenaDataItem> battles)
        {
            return battles.Where(b => b.Winner == "model_a" || b.Winner == "model_b" || b.Winner == "tie").ToList();
        }
    }

    public class ArenaDataItem
    {
        public string ModelA { get; set; }
        public string ModelB { get; set; }
        public string Winner { get; set; }
        public int ModelATier { get; set; }
        public int ModelBTier { get; set; }
    }
}