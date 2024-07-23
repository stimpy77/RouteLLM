using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;
using RouteLLM.Core;

namespace RouteLLM.Evaluations
{
    public abstract class Benchmark
    {
        protected ModelPair RoutedPair { get; set; }
        protected Dictionary<string, List<float>> Cache { get; set; }
        protected HashSet<string> OverwriteCache { get; set; }

        protected Benchmark(ModelPair routedPair, bool overwriteCache)
        {
            RoutedPair = routedPair;
            Cache = new Dictionary<string, List<float>>();
            OverwriteCache = new HashSet<string>();
            if (overwriteCache)
            {
                OverwriteCache.Add(routedPair.Strong);
                OverwriteCache.Add(routedPair.Weak);
            }
        }

        public abstract IEnumerable<(float Threshold, float Accuracy, Dictionary<string, int> ModelCounts, int Total)> Evaluate(
            Controller controller,
            string router,
            int numResults,
            bool overwriteRouterCache
        );

        public abstract float GetOptimalAccuracy(float strongPercent);

        public abstract float GetModelAccuracy(string model);

        protected void SaveCache(string router)
        {
            var cacheDir = Path.Combine("data", "cache");
            Directory.CreateDirectory(cacheDir);
            var cacheFile = Path.Combine(cacheDir, $"{GetType().Name}_{router}_cache.json");
            var jsonString = JsonSerializer.Serialize(Cache[router]);
            File.WriteAllText(cacheFile, jsonString);
        }

        protected void LoadCache(string router)
        {
            var cacheFile = Path.Combine("data", "cache", $"{GetType().Name}_{router}_cache.json");
            if (File.Exists(cacheFile))
            {
                var jsonString = File.ReadAllText(cacheFile);
                Cache[router] = JsonSerializer.Deserialize<List<float>>(jsonString);
            }
        }
    }

    public class MMLU : Benchmark
    {
        private List<Dictionary<string, string>> AllData { get; set; }

        public MMLU(List<string> mmluDomains, ModelPair routedPair, bool overwriteCache)
            : base(routedPair, overwriteCache)
        {
            AllData = LoadMMULData(mmluDomains);
        }

        public override IEnumerable<(float Threshold, float Accuracy, Dictionary<string, int> ModelCounts, int Total)> Evaluate(
            Controller controller,
            string router,
            int numResults,
            bool overwriteRouterCache)
        {
            if (!Cache.ContainsKey(router) || OverwriteCache.Contains(router) || overwriteRouterCache)
            {
                var strongWinRates = controller.BatchCalculateWinRate(AllData.Select(d => d["prompt"]).ToList(), router);
                Cache[router] = strongWinRates;
                SaveCache(router);
            }
            else
            {
                LoadCache(router);
                var strongWinRates = Cache[router];
            }

            var thresholds = GetThresholds(strongWinRates, numResults);

            foreach (var threshold in thresholds)
            {
                var (accuracy, modelCounts, total) = CalculateMetrics(strongWinRates, threshold);
                yield return (threshold, accuracy, modelCounts, total);
            }
        }

        public override float GetOptimalAccuracy(float strongPercent)
        {
            // TODO: Implement optimal accuracy calculation
            throw new NotImplementedException();
        }

        public override float GetModelAccuracy(string model)
        {
            var responses = AllData.Select(d => d[model]).ToList();
            var correctAnswers = AllData.Select(d => d["answer"]).ToList();
            return CalculateAccuracy(responses, correctAnswers);
        }

        private List<Dictionary<string, string>> LoadMMULData(List<string> mmluDomains)
        {
            var allData = new List<Dictionary<string, string>>();
            foreach (var domain in mmluDomains)
            {
                var filePath = Path.Combine("data", "mmlu", $"mmlu_{domain}.csv");
                var domainData = File.ReadAllLines(filePath)
                    .Skip(1) // Skip header
                    .Select(line => line.Split(','))
                    .Select(parts => new Dictionary<string, string>
                    {
                        ["prompt"] = parts[0],
                        ["answer"] = parts[1],
                        [RoutedPair.Strong] = parts[2],
                        [RoutedPair.Weak] = parts[3]
                    })
                    .ToList();
                allData.AddRange(domainData);
            }
            return allData;
        }

        private float CalculateAccuracy(List<string> responses, List<string> correctAnswers)
        {
            return BenchmarkUtils.CalculateAccuracy(responses, correctAnswers);
        }
    }

    public class GSM8K : Benchmark
    {
        private List<Dictionary<string, string>> AllData { get; set; }

        public GSM8K(ModelPair routedPair, bool overwriteCache)
            : base(routedPair, overwriteCache)
        {
            AllData = LoadGSM8KData();
        }

        public override IEnumerable<(float Threshold, float Accuracy, Dictionary<string, int> ModelCounts, int Total)> Evaluate(
            Controller controller,
            string router,
            int numResults,
            bool overwriteRouterCache)
        {
            if (!Cache.ContainsKey(router) || OverwriteCache.Contains(router) || overwriteRouterCache)
            {
                var strongWinRates = controller.BatchCalculateWinRate(AllData.Select(d => d["prompt"]).ToList(), router);
                Cache[router] = strongWinRates;
                SaveCache(router);
            }
            else
            {
                LoadCache(router);
                var strongWinRates = Cache[router];
            }

            var thresholds = GetThresholds(strongWinRates, numResults);

            foreach (var threshold in thresholds)
            {
                var (accuracy, modelCounts, total) = CalculateMetrics(strongWinRates, threshold);
                yield return (threshold, accuracy, modelCounts, total);
            }
        }

        public override float GetOptimalAccuracy(float strongPercent)
        {
            var strongAccuracy = GetModelAccuracy(RoutedPair.Strong);
            var weakAccuracy = GetModelAccuracy(RoutedPair.Weak);
            return strongPercent * strongAccuracy + (1 - strongPercent) * weakAccuracy;
        }

        public override float GetModelAccuracy(string model)
        {
            var responses = AllData.Select(d => d[model]).ToList();
            var correctAnswers = AllData.Select(d => d["answer"]).ToList();
            return CalculateAccuracy(responses, correctAnswers);
        }

        private List<Dictionary<string, string>> LoadGSM8KData()
        {
            var filePath = Path.Combine("data", "gsm8k", "gsm8k_responses.csv");
            return File.ReadAllLines(filePath)
                .Skip(1) // Skip header
                .Select(line => line.Split(','))
                .Select(parts => new Dictionary<string, string>
                {
                    ["prompt"] = parts[0],
                    ["answer"] = parts[1],
                    [RoutedPair.Strong] = parts[2],
                    [RoutedPair.Weak] = parts[3]
                })
                .ToList();
        }

        private float CalculateAccuracy(List<string> responses, List<string> correctAnswers)
        {
            return BenchmarkUtils.CalculateAccuracy(responses, correctAnswers);
        }
    }

    public class MTBench : Benchmark
    {
        private List<Dictionary<string, object>> AllData { get; set; }

        public MTBench(ModelPair routedPair, bool overwriteCache)
            : base(routedPair, overwriteCache)
        {
            AllData = LoadMTBenchData();
        }

        public override IEnumerable<(float Threshold, float Accuracy, Dictionary<string, int> ModelCounts, int Total)> Evaluate(
            Controller controller,
            string router,
            int numResults,
            bool overwriteRouterCache)
        {
            if (!Cache.ContainsKey(router) || OverwriteCache.Contains(router) || overwriteRouterCache)
            {
                var strongWinRates = controller.BatchCalculateWinRate(AllData.Select(d => (string)d["prompt"]).ToList(), router);
                Cache[router] = strongWinRates;
                SaveCache(router);
            }
            else
            {
                LoadCache(router);
                var strongWinRates = Cache[router];
            }

            var thresholds = BenchmarkUtils.GetThresholds(strongWinRates, numResults);

            foreach (var threshold in thresholds)
            {
                var (accuracy, modelCounts, total) = CalculateMetrics(strongWinRates, threshold);
                yield return (threshold, accuracy, modelCounts, total);
            }
        }

        public override float GetOptimalAccuracy(float strongPercent)
        {
            var strongAccuracy = GetModelAccuracy(RoutedPair.Strong);
            var weakAccuracy = GetModelAccuracy(RoutedPair.Weak);
            return strongPercent * strongAccuracy + (1 - strongPercent) * weakAccuracy;
        }

        public override float GetModelAccuracy(string model)
        {
            var scores = AllData.Select(d => (float)d[$"{model}_score"]).ToList();
            return scores.Average();
        }

        private List<Dictionary<string, object>> LoadMTBenchData()
        {
            var filePath = Path.Combine("data", "mt_bench", "mt_bench_responses.json");
            var jsonString = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<List<Dictionary<string, object>>>(jsonString);
        }

        private (float Accuracy, Dictionary<string, int> ModelCounts, int Total) CalculateMetrics(List<float> strongWinRates, float threshold)
        {
            var modelCounts = new Dictionary<string, int>
            {
                { RoutedPair.Strong, 0 },
                { RoutedPair.Weak, 0 }
            };

            float totalScore = 0;
            int total = strongWinRates.Count;

            for (int i = 0; i < total; i++)
            {
                if (strongWinRates[i] >= threshold)
                {
                    totalScore += (float)AllData[i][$"{RoutedPair.Strong}_score"];
                    modelCounts[RoutedPair.Strong]++;
                }
                else
                {
                    totalScore += (float)AllData[i][$"{RoutedPair.Weak}_score"];
                    modelCounts[RoutedPair.Weak]++;
                }
            }

            float accuracy = totalScore / total;
            return (accuracy, modelCounts, total);
        }
    }

    // Utility methods that can be used by multiple benchmark classes
    public static class BenchmarkUtils
    {
        public static List<float> GetThresholds(List<float> strongWinRates, int numResults)
        {
            var sortedRates = strongWinRates.OrderBy(r => r).ToList();
            var step = (float)sortedRates.Count / (numResults + 1);
            return Enumerable.Range(1, numResults)
                .Select(i => sortedRates[(int)(i * step)])
                .ToList();
        }

        public static float CalculateAccuracy(List<string> responses, List<string> correctAnswers)
        {
            if (responses.Count != correctAnswers.Count)
            {
                throw new ArgumentException("The number of responses and correct answers must be the same.");
            }

            int correct = 0;
            for (int i = 0; i < responses.Count; i++)
            {
                if (responses[i] == correctAnswers[i])
                {
                    correct++;
                }
            }

            return (float)correct / responses.Count;
        }
    }
}