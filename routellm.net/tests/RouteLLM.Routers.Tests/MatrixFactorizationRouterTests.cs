using System;
using System.Threading.Tasks;
using Xunit;
using Moq;
using TorchSharp;
using static TorchSharp.torch;
using RouteLLM.Routers;
using RouteLLM.Core;

namespace RouteLLM.Routers.Tests
{
    public class MatrixFactorizationRouterTests
    {
        [Fact]
        public async Task CalculateStrongWinRate_ReturnsValidProbability()
        {
            // Arrange
            var mockModel = new Mock<Module>();
            mockModel.Setup(m => m.forward(It.IsAny<Dictionary<string, Tensor>>()))
                .Returns(torch.tensor(0.7f));

            var router = new MatrixFactorizationRouter(
                "path/to/checkpoint",
                "gpt-4-1106-preview",
                "mixtral-8x7b-instruct-v0.1",
                model: mockModel.Object
            );

            // Act
            var result = await router.CalculateStrongWinRate("Test prompt");

            // Assert
            Assert.InRange(result, 0, 1);
            Assert.Equal(0.7f, result);
        }

        [Fact]
        public async Task Route_ReturnsCorrectModel()
        {
            // Arrange
            var mockModel = new Mock<Module>();
            mockModel.Setup(m => m.forward(It.IsAny<Dictionary<string, Tensor>>()))
                .Returns(torch.tensor(0.7f));

            var router = new MatrixFactorizationRouter(
                "path/to/checkpoint",
                "gpt-4-1106-preview",
                "mixtral-8x7b-instruct-v0.1",
                model: mockModel.Object
            );

            var modelPair = new ModelPair("gpt-4-1106-preview", "mixtral-8x7b-instruct-v0.1");

            // Act
            var resultStrong = await router.Route("Test prompt", 0.6f, modelPair);
            var resultWeak = await router.Route("Test prompt", 0.8f, modelPair);

            // Assert
            Assert.Equal("gpt-4-1106-preview", resultStrong);
            Assert.Equal("mixtral-8x7b-instruct-v0.1", resultWeak);
        }
    }
}