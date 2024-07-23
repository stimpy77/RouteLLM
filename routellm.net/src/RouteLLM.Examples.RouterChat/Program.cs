using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RouteLLM.Client;
using RouteLLM.Core;

class Program
{
    static async Task Main(string[] args)
    {
        var client = new RouteLLMClient("http://localhost:5000/v1", "your_api_key");
        var router = "random";
        var threshold = 0.5f;
        var temperature = 0.7f;

        Console.WriteLine("Welcome to RouteLLM Chat! Type 'exit' to quit.");
        Console.WriteLine($"Using router: {router} with threshold: {threshold}");

        var history = new List<(string, string)>();

        while (true)
        {
            Console.Write("You: ");
            var userInput = Console.ReadLine();

            if (userInput.ToLower() == "exit")
                break;

            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "system", Content = "You are a helpful AI assistant." }
            };

            foreach (var (human, assistant) in history)
            {
                messages.Add(new ChatMessage { Role = "user", Content = human });
                messages.Add(new ChatMessage { Role = "assistant", Content = assistant });
            }

            messages.Add(new ChatMessage { Role = "user", Content = userInput });

            var request = new CompletionRequest
            {
                Model = $"router-{router}-{threshold}",
                Messages = messages,
                Temperature = temperature
            };

            try
            {
                var response = await client.CreateChatCompletion(request);
                var assistantResponse = response.Choices[0].Message.Content;
                Console.WriteLine($"Assistant: {assistantResponse}");

                history.Add((userInput, assistantResponse));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
        }
    }
}