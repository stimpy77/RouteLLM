using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Moq;
using Xunit;
using RouteLLM.Core;

namespace RouteLLM.Core.Tests
{
    public class ControllerTests
    {
        [Fact]
        public async Task Route_ReturnsCorrectModel()
        {
            // Arrange
            var mockRouter = new Mock<IRouter>();
            mockRouter.Setup(r => r.CalculateStrongWinRate(It.IsAny<string>()))
                .ReturnsAsync(0.7f);

            var routerFactory = new Mock<IRouterFactory>();
            routerFactory.Setup(f => f.CreateRouter("test_router"))
                .Returns(mockRouter.Object);

            var controller = new Controller(
                new[] { "test_router" },
                "strong_model",
                "weak_model",
                routerFactory.Object
            );

            // Act
            var result = await controller.Route("Test prompt", "test_router", 0.5f);

            // Assert
            Assert.Equal("strong_model", result);
        }

        [Fact]
        public async Task BatchCalculateWinRate_ReturnsCorrectResults()
        {
            // Arrange
            var mockRouter = new Mock<IRouter>();
            mockRouter.Setup(r => r.CalculateStrongWinRate(It.IsAny<string>()))
                .ReturnsAsync(0.7f);

            var routerFactory = new Mock<IRouterFactory>();
            routerFactory.Setup(f => f.CreateRouter("test_router"))
                .Returns(mockRouter.Object);

            var controller = new Controller(
                new[] { "test_router" },
                "strong_model",
                "weak_model",
                routerFactory.Object
            );

            var prompts = new List<string> { "Prompt 1", "Prompt 2" };

            // Act
            var results = await controller.BatchCalculateWinRate(prompts, "test_router");

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Equal(0.7f, r));
        }

        [Fact]
        public async Task Completion_ReturnsCorrectResponse()
        {
            // Arrange
            var mockRouter = new Mock<IRouter>();
            mockRouter.Setup(r => r.CalculateStrongWinRate(It.IsAny<string>()))
                .ReturnsAsync(0.7f);

            var routerFactory = new Mock<IRouterFactory>();
            routerFactory.Setup(f => f.CreateRouter("test_router"))
                .Returns(mockRouter.Object);

            var controller = new Controller(
                new[] { "test_router" },
                "strong_model",
                "weak_model",
                routerFactory.Object
            );

            var request = new CompletionRequest
            {
                Model = "router-test_router-0.5",
                Messages = new List<ChatMessage>
                {
                    new ChatMessage { Role = "user", Content = "Hello" }
                }
            };

            // Act
            var response = await controller.Completion(request);

            // Assert
            Assert.NotNull(response);
            Assert.Equal("strong_model", response.Model);
            Assert.NotEmpty(response.Choices);
        }
    }
}