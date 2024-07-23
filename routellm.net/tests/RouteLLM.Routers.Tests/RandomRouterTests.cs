using System;
using System.Threading.Tasks;
using Xunit;
using RouteLLM.Routers;
using RouteLLM.Core;

namespace RouteLLM.Routers.Tests
{
    public class RandomRouterTests
    {
        [Fact]
        public async Task CalculateStrongWinRate_ReturnsValueBetweenZeroAndOne()
        {
            // Arrange
            var router = new RandomRouter();

            // Act
            var result = await router.CalculateStrongWinRate("Test prompt");

            // Assert
            Assert.InRange(result, 0, 1);
        }

        [Fact]
        public async Task Route_ReturnsStrongModelWhenWinRateAboveThreshold()
        {
            // Arrange
            var router = new RandomRouter();
            var modelPair = new ModelPair("strong_model", "weak_model");
            const float threshold = 0.5f;

            // Act
            var result = await router.Route("Test prompt", threshold, modelPair);

            // Assert
            Assert.True(result == "strong_model" || result == "weak_model");
        }

        [Fact]
        public async Task Route_ReturnsExpectedDistribution()
        {
            // Arrange
            var router = new RandomRouter();
            var modelPair = new ModelPair("strong_model", "weak_model");
            const float threshold = 0.5f;
            const int iterations = 10000;
            int strongCount = 0;

            // Act
            for (int i = 0; i < iterations; i++)
            {
                var result = await router.Route("Test prompt", threshold, modelPair);
                if (result == "strong_model")
                {
                    strongCount++;
                }
            }

            // Assert
            var strongPercentage = (double)strongCount / iterations;
            Assert.InRange(strongPercentage, 0.45, 0.55); // Allow for some randomness
        }
    }
}
