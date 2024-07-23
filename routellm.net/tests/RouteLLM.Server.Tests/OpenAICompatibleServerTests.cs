using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;
using RouteLLM.Core;
using System.Text.Json;

namespace RouteLLM.Server.Tests
{
    public class OpenAICompatibleServerTests : IClassFixture<WebApplicationFactory<Program>>
    {
        private readonly WebApplicationFactory<Program> _factory;

        public OpenAICompatibleServerTests(WebApplicationFactory<Program> factory)
        {
            _factory = factory;
        }

        [Fact]
        public async Task CreateChatCompletion_ReturnsSuccessStatusCode()
        {
            // Arrange
            var client = _factory.CreateClient();
            var request = new CompletionRequest
            {
                Model = "router-random-0.5",
                Messages = new List<ChatMessage>
                {
                    new ChatMessage { Role = "user", Content = "Hello, how are you?" }
                }
            };

            // Act
            var response = await client.PostAsJsonAsync("/v1/chat/completions", request);

            // Assert
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var completionResponse = JsonSerializer.Deserialize<CompletionResponse>(content);
            Assert.NotNull(completionResponse);
            Assert.NotEmpty(completionResponse.Choices);
        }

        [Fact]
        public async Task ListModels_ReturnsSuccessStatusCode()
        {
            // Arrange
            var client = _factory.CreateClient();

            // Act
            var response = await client.GetAsync("/v1/models");

            // Assert
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var modelListResponse = JsonSerializer.Deserialize<ModelListResponse>(content);
            Assert.NotNull(modelListResponse);
            Assert.NotEmpty(modelListResponse.Data);
        }

        [Fact]
        public async Task RetrieveModel_ReturnsSuccessStatusCode()
        {
            // Arrange
            var client = _factory.CreateClient();

            // Act
            var response = await client.GetAsync("/v1/models/router-random");

            // Assert
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            var modelResponse = JsonSerializer.Deserialize<ModelResponse>(content);
            Assert.NotNull(modelResponse);
            Assert.Equal("router-random", modelResponse.Id);
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
