using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Moq;
using Xunit;
using RouteLLM.Core;
using RouteLLM.Evaluations;

namespace RouteLLM.Evaluations.Tests
{
    public class EvaluatorTests
    {
        [Fact]
        public async Task EvaluateRouter_ReturnsExpectedResults()
        {
            // Arrange
            var mockController = new Mock<Controller>(MockBehavior.Strict);
            mockController.Setup(c => c.BatchCalculateWinRate(It.IsAny<List<string>>(), It.IsAny<string>()))
                .ReturnsAsync(new List<float> { 0.7f, 0.8f, 0.6f });

            var evaluator = new Evaluator(mockController.Object);

            // Act
            var results = await evaluator.EvaluateRouter("test_router", new List<string> { "prompt1", "prompt2", "prompt3" });

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);
            Assert.All(results, r => Assert.InRange(r, 0, 1));
        }

        [Fact]
        public async Task EvaluateRouterWithMetrics_ReturnsExpectedMetrics()
        {
            // Arrange
            var mockController = new Mock<Controller>(MockBehavior.Strict);
            mockController.Setup(c => c.BatchCalculateWinRate(It.IsAny<List<string>>(), It.IsAny<string>()))
                .ReturnsAsync(new List<float> { 0.7f, 0.8f, 0.6f });

            var evaluator = new Evaluator(mockController.Object);

            // Act
            var results = await evaluator.EvaluateRouterWithMetrics("test_router", new List<string> { "prompt1", "prompt2", "prompt3" }, 0.75f);

            // Assert
            Assert.NotNull(results);
            Assert.Contains("win_rates", results.Keys);
            Assert.Contains("average_win_rate", results.Keys);
            Assert.Contains("strong_model_percentage", results.Keys);
            Assert.Contains("median_win_rate", results.Keys);
        }

        [Fact]
        public async Task CompareRouters_ReturnsResultsForAllRouters()
        {
            // Arrange
            var mockController = new Mock<Controller>(MockBehavior.Strict);
            mockController.Setup(c => c.BatchCalculateWinRate(It.IsAny<List<string>>(), It.IsAny<string>()))
                .ReturnsAsync(new List<float> { 0.7f, 0.8f, 0.6f });

            var evaluator = new Evaluator(mockController.Object);

            // Act
            var results = await evaluator.CompareRouters(new List<string> { "router1", "router2" }, new List<string> { "prompt1", "prompt2", "prompt3" }, 0.75f);

            // Assert
            Assert.NotNull(results);
            Assert.Equal(2, results.Count);
            Assert.Contains("router1", results.Keys);
            Assert.Contains("router2", results.Keys);
        }
    }
}