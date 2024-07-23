using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using RouteLLM.Core;

namespace RouteLLM.Client
{
    public class RouteLLMClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public RouteLLMClient(string baseUrl, string apiKey)
        {
            _baseUrl = baseUrl;
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        }

        public async Task<CompletionResponse> CreateChatCompletion(CompletionRequest request)
        {
            var response = await SendRequest<CompletionResponse>("chat/completions", request);
            return response;
        }

        public async Task<CompletionResponse> CreateCompletion(CompletionRequest request)
        {
            var response = await SendRequest<CompletionResponse>("completions", request);
            return response;
        }

        public async Task<ModelListResponse> ListModels()
        {
            var response = await SendRequest<ModelListResponse>("models", HttpMethod.Get);
            return response;
        }

        public async Task<ModelResponse> RetrieveModel(string model)
        {
            var response = await SendRequest<ModelResponse>($"models/{model}", HttpMethod.Get);
            return response;
        }

        private async Task<T> SendRequest<T>(string endpoint, object body = null, HttpMethod method = null)
        {
            var requestMessage = new HttpRequestMessage(method ?? HttpMethod.Post, $"{_baseUrl}/{endpoint}");

            if (body != null)
            {
                var json = JsonSerializer.Serialize(body);
                requestMessage.Content = new StringContent(json, Encoding.UTF8, "application/json");
            }

            var response = await _httpClient.SendAsync(requestMessage);
            response.EnsureSuccessStatusCode();

            var content = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<T>(content);
        }
    }

    public class ModelListResponse
    {
        public List<string> Data { get; set; }
    }

    public class ModelResponse
    {
        public string Id { get; set; }
        public string Object { get; set; }
        public long Created { get; set; }
        public string OwnedBy { get; set; }
    }
}
