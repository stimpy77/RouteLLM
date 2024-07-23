using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RouteLLM.Core;
using RouteLLM.Routers;

namespace RouteLLM.Evaluations
{
    public class Evaluator
    {
        private readonly Controller controller;

        public Evaluator(Controller controller)
        {
            this.controller = controller;
        }

        public async Task<Dictionary<string, float>> EvaluateRouter(string router, List<string> prompts, float threshold)
        {
            var results = new Dictionary<string, float>();
            var winRates = await controller.BatchCalculateWinRate(prompts, router);

            float totalWinRate = 0;
            int strongModelCount = 0;

            for (int i = 0; i < prompts.Count; i++)
            {
                float winRate = winRates[i];
                totalWinRate += winRate;
                if (winRate >= threshold)
                {
                    strongModelCount++;
                }
            }

            results["average_win_rate"] = totalWinRate / prompts.Count;
            results["strong_model_percentage"] = (float)strongModelCount / prompts.Count;

            return results;
        }

        public async Task<Dictionary<string, object>> EvaluateRouterWithMetrics(string router, List<string> prompts, float threshold)
        {
            var results = new Dictionary<string, object>();
            var winRates = await controller.BatchCalculateWinRate(prompts, router);

            results["win_rates"] = winRates;
            results["average_win_rate"] = winRates.Average();
            results["strong_model_percentage"] = (float)winRates.Count(wr => wr >= threshold) / prompts.Count;
            results["median_win_rate"] = Median(winRates);

            return results;
        }

        public async Task<Dictionary<string, object>> CompareRouters(List<string> routers, List<string> prompts, float threshold)
        {
            var results = new Dictionary<string, object>();

            foreach (var router in routers)
            {
                results[router] = await EvaluateRouterWithMetrics(router, prompts, threshold);
            }

            return results;
        }

        public async Task<Dictionary<string, object>> EvaluateMMLU(string router, List<string> mmluDomains, float threshold)
        {
            var benchmark = new MMLU(mmluDomains, controller.ModelPair, false);
            var results = benchmark.Evaluate(controller, router, 1, false).First();
            return new Dictionary<string, object>
            {
                ["accuracy"] = results.Accuracy,
                ["model_counts"] = results.ModelCounts,
                ["total"] = results.Total,
                ["optimal_accuracy"] = benchmark.GetOptimalAccuracy(results.ModelCounts[controller.ModelPair.Strong] / (float)results.Total)
            };
        }

        public async Task<Dictionary<string, object>> EvaluateGSM8K(string router, float threshold)
        {
            var benchmark = new GSM8K(controller.ModelPair, false);
            var results = benchmark.Evaluate(controller, router, 1, false).First();
            return new Dictionary<string, object>
            {
                ["accuracy"] = results.Accuracy,
                ["model_counts"] = results.ModelCounts,
                ["total"] = results.Total,
                ["optimal_accuracy"] = benchmark.GetOptimalAccuracy(results.ModelCounts[controller.ModelPair.Strong] / (float)results.Total)
            };
        }

        public async Task<Dictionary<string, object>> EvaluateMTBench(string router, float threshold)
        {
            var benchmark = new MTBench(controller.ModelPair, false);
            var results = benchmark.Evaluate(controller, router, 1, false).First();
            return new Dictionary<string, object>
            {
                ["score"] = results.Accuracy,
                ["model_counts"] = results.ModelCounts,
                ["total"] = results.Total,
                ["optimal_score"] = benchmark.GetOptimalAccuracy(results.ModelCounts[controller.ModelPair.Strong] / (float)results.Total)
            };
        }

        private float Median(List<float> values)
        {
            var sortedValues = values.OrderBy(v => v).ToList();
            int count = sortedValues.Count;
            if (count % 2 == 0)
            {
                return (sortedValues[count / 2 - 1] + sortedValues[count / 2]) / 2;
            }
            else
            {
                return sortedValues[count / 2];
            }
        }
    }
}