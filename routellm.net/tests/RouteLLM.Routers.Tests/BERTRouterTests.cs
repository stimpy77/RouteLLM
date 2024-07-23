using System;
using System.Threading.Tasks;
using Xunit;
using Moq;
using Microsoft.ML;
using RouteLLM.Routers;
using RouteLLM.Core;

namespace RouteLLM.Routers.Tests
{
    public class BERTRouterTests
    {
        [Fact]
        public async Task CalculateStrongWinRate_ReturnsValidProbability()
        {
            // Arrange
            var mockPredictionEngine = new Mock<PredictionEngine<BERTInputData, BERTOutputData>>();
            mockPredictionEngine.Setup(pe => pe.Predict(It.IsAny<BERTInputData>()))
                .Returns(new BERTOutputData { Probabilities = new float[] { 0.2f, 0.3f, 0.5f } });

            var router = new BERTRouter("path/to/checkpoint", predictionEngine: mockPredictionEngine.Object);

            // Act
            var result = await router.CalculateStrongWinRate("Test prompt");

            // Assert
            Assert.InRange(result, 0, 1);
            Assert.Equal(0.2f, result, 3); // 1 - (0.3 + 0.5) = 0.2
        }

        [Fact]
        public async Task Route_ReturnsCorrectModel()
        {
            // Arrange
            var mockPredictionEngine = new Mock<PredictionEngine<BERTInputData, BERTOutputData>>();
            mockPredictionEngine.Setup(pe => pe.Predict(It.IsAny<BERTInputData>()))
                .Returns(new BERTOutputData { Probabilities = new float[] { 0.2f, 0.3f, 0.5f } });

            var router = new BERTRouter("path/to/checkpoint", predictionEngine: mockPredictionEngine.Object);
            var modelPair = new ModelPair("strong_model", "weak_model");

            // Act
            var resultStrong = await router.Route("Test prompt", 0.1f, modelPair);
            var resultWeak = await router.Route("Test prompt", 0.3f, modelPair);

            // Assert
            Assert.Equal("strong_model", resultStrong);
            Assert.Equal("weak_model", resultWeak);
        }
    }
}