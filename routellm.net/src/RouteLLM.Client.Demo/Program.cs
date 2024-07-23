using System;
using System.Threading.Tasks;
using RouteLLM.Client;
using RouteLLM.Core;

class Program
{
    static async Task Main(string[] args)
    {
        var client = new RouteLLMClient("http://localhost:5000/v1", "test_api_key");

        try
        {
            // List models
            var models = await client.ListModels();
            Console.WriteLine("Available models:");
            foreach (var model in models.Data)
            {
                Console.WriteLine($"- {model}");
            }

            // Create a chat completion
            var chatRequest = new CompletionRequest
            {
                Model = "router-random-0.5",
                Messages = new List<ChatMessage>
                {
                    new ChatMessage { Role = "user", Content = "Hello, how are you?" }
                }
            };

            var chatResponse = await client.CreateChatCompletion(chatRequest);
            Console.WriteLine($"\nChat Completion Response: {chatResponse.Choices[0].Message.Content}");

            // Retrieve model information
            var modelInfo = await client.RetrieveModel("router-random");
            Console.WriteLine($"\nModel Information:");
            Console.WriteLine($"ID: {modelInfo.Id}");
            Console.WriteLine($"Created: {DateTimeOffset.FromUnixTimeSeconds(modelInfo.Created).ToString("g")}");
            Console.WriteLine($"Owned By: {modelInfo.OwnedBy}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }
}