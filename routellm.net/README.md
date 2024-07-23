# RouteLLM.NET

RouteLLM.NET is a C# port of the RouteLLM project, a framework for serving and evaluating large language model routers.

## Project Structure

The solution is organized into the following projects:

1. `RouteLLM.Core`: Contains core interfaces and classes used across the solution.
2. `RouteLLM.Routers`: Implements various router algorithms.
3. `RouteLLM.Server`: Provides an OpenAI-compatible API server.
4. `RouteLLM.Evaluations`: Contains benchmark implementations and evaluation logic.
5. `RouteLLM.Client`: Implements an OpenAI-compatible client for testing.
6. `RouteLLM.Client.Demo`: Demonstrates usage of the RouteLLM client.

Test projects:
- `RouteLLM.Core.Tests`
- `RouteLLM.Routers.Tests`
- `RouteLLM.Server.Tests`
- `RouteLLM.Evaluations.Tests`

## Getting Started

### Prerequisites

- .NET 8.0 SDK or later
- Visual Studio 2022 or JetBrains Rider (recommended)

### Building the Project

1. Clone the repository:
   ```
   git clone https://github.com/your-username/routellm.net.git
   cd routellm.net
   ```

2. Build the solution:
   ```
   dotnet build
   ```

3. Run the tests:
   ```
   dotnet test
   ```

### Running the Server

To start the RouteLLM server:

```
cd src/RouteLLM.Server
dotnet run
```

The server will start on `http://localhost:5000` by default.

### Using the Client

Here's a basic example of how to use the RouteLLM client:

```csharp
using RouteLLM.Client;
using RouteLLM.Core;

var client = new RouteLLMClient("http://localhost:5000/v1", "your_api_key");

var chatRequest = new CompletionRequest
{
    Model = "router-random-0.5",
    Messages = new List<ChatMessage>
    {
        new ChatMessage { Role = "user", Content = "Hello, how are you?" }
    }
};

var chatResponse = await client.CreateChatCompletion(chatRequest);
Console.WriteLine($"Response: {chatResponse.Choices[0].Message.Content}");
```

## Available Routers

- RandomRouter
- CausalLLMRouter
- BERTRouter
- SWRankingRouter
- MatrixFactorizationRouter

## Benchmarks

The following benchmarks are implemented:

- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math 8K)
- MT-Bench (Machine Translation Benchmark)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.