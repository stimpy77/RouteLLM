using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using Moq;
using RouteLLM.Routers;
using RouteLLM.Core;

namespace RouteLLM.Routers.Tests
{
    public class SWRankingRouterTests
    {
        [Fact]
        public async Task CalculateStrongWinRate_ReturnsValidProbability()
        {
            // Arrange
            var mockUtils = new Mock<ISWRankingUtils>();
            mockUtils.Setup(u => u.GetEmbedding(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(new float[] { 1, 2, 3 });
            mockUtils.Setup(u => u.ComputeSimilarities(It.IsAny<float[]>(), It.IsAny<float[,]>()))
                .Returns(new float[] { 0.5f, 0.7f, 0.3f });
            mockUtils.Setup(u => u.ComputeEloMleWithTie(It.IsAny<List<ArenaDataItem>>(), It.IsAny<float[]>()))
                .Returns(new Dictionary<string, float> { { "strong_model", 1600 }, { "weak_model", 1400 } });

            var router = new SWRankingRouter(
                new List<string> { "path/to/arena/battles" },
                new List<string> { "path/to/arena/embeddings" },
                "strong_model",
                "weak_model",
                swRankingUtils: mockUtils.Object
            );

            // Act
            var result = await router.CalculateStrongWinRate("Test prompt");

            // Assert
            Assert.InRange(result, 0, 1);
        }

        [Fact]
        public async Task Route_ReturnsCorrectModel()
        {
            // Arrange
            var mockUtils = new Mock<ISWRankingUtils>();
            mockUtils.Setup(u => u.GetEmbedding(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(new float[] { 1, 2, 3 });
            mockUtils.Setup(u => u.ComputeSimilarities(It.IsAny<float[]>(), It.IsAny<float[,]>()))
                .Returns(new float[] { 0.5f, 0.7f, 0.3f });
            mockUtils.Setup(u => u.ComputeEloMleWithTie(It.IsAny<List<ArenaDataItem>>(), It.IsAny<float[]>()))
                .Returns(new Dictionary<string, float> { { "strong_model", 1600 }, { "weak_model", 1400 } });

            var router = new SWRankingRouter(
                new List<string> { "path/to/arena/battles" },
                new List<string> { "path/to/arena/embeddings" },
                "strong_model",
                "weak_model",
                swRankingUtils: mockUtils.Object
            );

            var modelPair = new ModelPair("strong_model", "weak_model");

            // Act
            var resultStrong = await router.Route("Test prompt", 0.6f, modelPair);
            var resultWeak = await router.Route("Test prompt", 0.8f, modelPair);

            // Assert
            Assert.Equal("strong_model", resultStrong);
            Assert.Equal("weak_model", resultWeak);
        }
    }
}