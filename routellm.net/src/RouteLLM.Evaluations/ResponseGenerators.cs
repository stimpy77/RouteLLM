using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RouteLLM.Core;
using System.Text.Json;

namespace RouteLLM.Evaluations
{
    public abstract class ResponseGenerator
    {
        protected Controller Controller { get; }
        protected ModelPair RoutedPair { get; }

        protected ResponseGenerator(Controller controller, ModelPair routedPair)
        {
            Controller = controller;
            RoutedPair = routedPair;
        }

        public abstract Task GenerateResponses();
    }

    public class GSM8KResponseGenerator : ResponseGenerator
    {
        public GSM8KResponseGenerator(Controller controller, ModelPair routedPair) : base(controller, routedPair) { }

        public override async Task GenerateResponses()
        {
            var testData = LoadGSM8KTestData();
            var weakResponses = await GenerateModelResponses(RoutedPair.Weak, testData);
            var strongResponses = await GenerateModelResponses(RoutedPair.Strong, testData);

            var results = testData.Zip(weakResponses, strongResponses, (prompt, weak, strong) => new
            {
                prompt,
                weak_response = weak,
                strong_response = strong
            }).ToList();

            SaveResponses(results);
        }

        private List<string> LoadGSM8KTestData()
        {
            var filePath = Path.Combine("data", "gsm8k", "test.jsonl");
            return File.ReadAllLines(filePath)
                .Select(line => JsonSerializer.Deserialize<Dictionary<string, string>>(line)["question"])
                .ToList();
        }

        private async Task<List<string>> GenerateModelResponses(string model, List<string> prompts)
        {
            var responses = new List<string>();
            foreach (var prompt in prompts)
            {
                var response = await Controller.Completion(new CompletionRequest
                {
                    Model = model,
                    Messages = new List<ChatMessage>
                    {
                        new ChatMessage { Role = "user", Content = prompt }
                    }
                });
                responses.Add(response.Choices[0].Message.Content);
            }
            return responses;
        }

        private void SaveResponses(List<dynamic> results)
        {
            var outputPath = Path.Combine("data", "gsm8k", "gsm8k_responses.csv");
            using (var writer = new StreamWriter(outputPath))
            {
                writer.WriteLine("prompt,weak_response,strong_response");
                foreach (var result in results)
                {
                    writer.WriteLine($"{result.prompt},{result.weak_response},{result.strong_response}");
                }
            }
        }
    }

    public class MMLUResponseGenerator : ResponseGenerator
    {
        private readonly List<string> Domains;

        public MMLUResponseGenerator(Controller controller, ModelPair routedPair, List<string> domains) : base(controller, routedPair)
        {
            Domains = domains;
        }

        public override async Task GenerateResponses()
        {
            foreach (var domain in Domains)
            {
                var testData = LoadMMULTestData(domain);
                var weakResponses = await GenerateModelResponses(RoutedPair.Weak, testData);
                var strongResponses = await GenerateModelResponses(RoutedPair.Strong, testData);

                var results = testData.Zip(weakResponses, strongResponses, (data, weak, strong) => new
                {
                    prompt = data.prompt,
                    answer = data.answer,
                    weak_response = weak,
                    strong_response = strong
                }).ToList();

                SaveResponses(domain, results);
            }
        }

        private List<(string prompt, string answer)> LoadMMULTestData(string domain)
        {
            var filePath = Path.Combine("data", "mmlu", $"{domain}_test.csv");
            return File.ReadAllLines(filePath)
                .Skip(1) // Skip header
                .Select(line =>
                {
                    var parts = line.Split(',');
                    return (prompt: parts[0], answer: parts[1]);
                })
                .ToList();
        }

        private async Task<List<string>> GenerateModelResponses(string model, List<(string prompt, string answer)> testData)
        {
            var responses = new List<string>();
            foreach (var (prompt, _) in testData)
            {
                var response = await Controller.Completion(new CompletionRequest
                {
                    Model = model,
                    Messages = new List<ChatMessage>
                    {
                        new ChatMessage { Role = "user", Content = prompt }
                    }
                });
                responses.Add(response.Choices[0].Message.Content);
            }
            return responses;
        }

        private void SaveResponses(string domain, List<dynamic> results)
        {
            var outputPath = Path.Combine("data", "mmlu", $"mmlu_{domain}_responses.csv");
            using (var writer = new StreamWriter(outputPath))
            {
                writer.WriteLine("prompt,answer,weak_response,strong_response");
                foreach (var result in results)
                {
                    writer.WriteLine($"{result.prompt},{result.answer},{result.weak_response},{result.strong_response}");
                }
            }
        }
    }
}
