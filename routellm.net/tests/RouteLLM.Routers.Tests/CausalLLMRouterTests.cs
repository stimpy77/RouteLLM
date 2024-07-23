using System;
using System.Threading.Tasks;
using Xunit;
using Moq;
using RouteLLM.Routers;
using RouteLLM.Core;

namespace RouteLLM.Routers.Tests
{
    public class CausalLLMRouterTests
    {
        [Fact]
        public async Task CalculateStrongWinRate_ReturnsValidProbability()
        {
            // Arrange
            var mockClassifier = new Mock<CausalLLMClassifier>();
            mockClassifier.Setup(c => c.Predict(It.IsAny<string>())).Returns(0.7f);

            var router = new CausalLLMRouter("path/to/checkpoint", classifier: mockClassifier.Object);

            // Act
            var result = await router.CalculateStrongWinRate("Test prompt");

            // Assert
            Assert.InRange(result, 0, 1);
            Assert.Equal(0.3f, result, 3); // 1 - 0.7 = 0.3
        }

        [Fact]
        public async Task Route_ReturnsCorrectModel()
        {
            // Arrange
            var mockClassifier = new Mock<CausalLLMClassifier>();
            mockClassifier.Setup(c => c.Predict(It.IsAny<string>())).Returns(0.7f);

            var router = new CausalLLMRouter("path/to/checkpoint", classifier: mockClassifier.Object);
            var modelPair = new ModelPair("strong_model", "weak_model");

            // Act
            var resultStrong = await router.Route("Test prompt", 0.2f, modelPair);
            var resultWeak = await router.Route("Test prompt", 0.4f, modelPair);

            // Assert
            Assert.Equal("strong_model", resultStrong);
            Assert.Equal("weak_model", resultWeak);
        }
    }
}