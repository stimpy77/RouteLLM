using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;

namespace RouteLLM.Core
{
    public class Controller
    {
        private ModelPair modelPair;
        private Dictionary<string, IRouter> routers;
        private string apiBase;
        private string apiKey;
        private Dictionary<string, Dictionary<string, int>> modelCounts;
        private bool progressBar;
        private readonly HttpClient httpClient;

        public Controller(
            List<string> routers,
            string strongModel,
            string weakModel,
            Dictionary<string, Dictionary<string, object>> config = null,
            string apiBase = null,
            string apiKey = null,
            bool progressBar = false)
        {
            this.modelPair = new ModelPair(strongModel, weakModel);
            this.routers = routers.ToDictionary(
                r => r,
                r => RouterFactory.CreateRouter(r, config?[r])
            );
            this.apiBase = apiBase;
            this.apiKey = apiKey;
            this.modelCounts = new Dictionary<string, Dictionary<string, int>>();
            this.progressBar = progressBar;
            this.httpClient = new HttpClient();
            if (!string.IsNullOrEmpty(apiBase))
            {
                this.httpClient.BaseAddress = new Uri(apiBase);
            }
            if (!string.IsNullOrEmpty(apiKey))
            {
                this.httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
            }
        }

        private void ValidateRouterThreshold(string router, float threshold)
        {
            if (string.IsNullOrEmpty(router) || !routers.ContainsKey(router))
            {
                throw new RoutingError($"Invalid router {router}. Available routers are {string.Join(", ", routers.Keys)}.");
            }

            if (threshold < 0 || threshold > 1)
            {
                throw new RoutingError($"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0.");
            }
        }

        private (string router, float threshold) ParseModelName(string model)
        {
            var parts = model.Split('-');
            if (parts.Length != 3 || parts[0] != "router")
            {
                throw new RoutingError($"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold].");
            }

            if (!float.TryParse(parts[2], out float threshold))
            {
                throw new RoutingError($"Threshold {parts[2]} must be a float.");
            }

            return (parts[1], threshold);
        }

        private async Task<string> GetRoutedModelForCompletion(List<Dictionary<string, string>> messages, string router, float threshold)
        {
            string prompt = messages.Last()["content"];
            string routedModel = await routers[router].Route(prompt, threshold, modelPair);

            if (!modelCounts.ContainsKey(router))
            {
                modelCounts[router] = new Dictionary<string, int>();
            }
            if (!modelCounts[router].ContainsKey(routedModel))
            {
                modelCounts[router][routedModel] = 0;
            }
            modelCounts[router][routedModel]++;

            return routedModel;
        }

        public async Task<string> Route(string prompt, string router, float threshold)
        {
            ValidateRouterThreshold(router, threshold);
            return await routers[router].Route(prompt, threshold, modelPair);
        }

        public async Task<List<float>> BatchCalculateWinRate(List<string> prompts, string router)
        {
            ValidateRouterThreshold(router, 0);
            var routerInstance = routers[router];
            var results = new List<float>();

            if (routerInstance is Router routerBase && routerBase.NoParallel)
            {
                foreach (var prompt in prompts)
                {
                    results.Add(await routerInstance.CalculateStrongWinRate(prompt));
                }
            }
            else
            {
                results = await Task.WhenAll(prompts.Select(p => routerInstance.CalculateStrongWinRate(p)));
            }

            return results;
        }

        public async Task<CompletionResponse> Completion(CompletionRequest request)
        {
            string router = null;
            float threshold = 0;

            if (request.Model != null && request.Model.StartsWith("router-"))
            {
                (router, threshold) = ParseModelName(request.Model);
            }
            else
            {
                router = request.Router;
                threshold = request.Threshold ?? 0;
            }

            ValidateRouterThreshold(router, threshold);
            string routedModel = await GetRoutedModelForCompletion(request.Messages, router, threshold);

            // Make the API call to the language model
            var apiRequest = new
            {
                model = routedModel,
                messages = request.Messages,
                temperature = request.Temperature,
                max_tokens = request.MaxTokens
                // Add other parameters as needed
            };

            var response = await httpClient.PostAsJsonAsync("/v1/chat/completions", apiRequest);
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var completionResponse = JsonSerializer.Deserialize<CompletionResponse>(content);

            return completionResponse;
        }

        public async Task<CompletionResponse> ACompletion(CompletionRequest request)
        {
            // The async version can be the same as the sync version in this case
            return await Completion(request);
        }
    }

    public class CompletionRequest
    {
        public string Model { get; set; }
        public List<Dictionary<string, string>> Messages { get; set; }
        public string Router { get; set; }
        public float? Threshold { get; set; }
        public float Temperature { get; set; }
        public int? MaxTokens { get; set; }
        // Add other properties as needed
    }

    public class CompletionResponse
    {
        public string Id { get; set; }
        public string Object { get; set; }
        public long Created { get; set; }
        public string Model { get; set; }
        public List<Choice> Choices { get; set; }
        public Usage Usage { get; set; }
    }

    public class Choice
    {
        public int Index { get; set; }
        public Message Message { get; set; }
        public string FinishReason { get; set; }
    }

    public class Message
    {
        public string Role { get; set; }
        public string Content { get; set; }
    }

    public class Usage
    {
        public int PromptTokens { get; set; }
        public int CompletionTokens { get; set; }
        public int TotalTokens { get; set; }
    }
}