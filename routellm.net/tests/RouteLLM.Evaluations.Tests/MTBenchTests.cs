using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using Moq;
using RouteLLM.Core;
using RouteLLM.Evaluations;

namespace RouteLLM.Evaluations.Tests
{
    public class MTBenchTests
    {
        [Fact]
        public void Evaluate_ReturnsExpectedResults()
        {
            // Arrange
            var mockController = new Mock<Controller>();
            mockController.Setup(c => c.BatchCalculateWinRate(It.IsAny<List<string>>(), It.IsAny<string>()))
                .ReturnsAsync(new List<float> { 0.7f, 0.8f, 0.6f, 0.9f, 0.5f });

            var mtbench = new MTBench(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var results = mtbench.Evaluate(mockController.Object, "test_router", 3, false).ToList();

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);
            Assert.All(results, r =>
            {
                Assert.InRange(r.Threshold, 0, 1);
                Assert.InRange(r.Accuracy, 0, 10); // MT-Bench typically uses scores from 0 to 10
                Assert.NotNull(r.ModelCounts);
                Assert.Equal(5, r.Total);
            });
        }

        [Fact]
        public void GetOptimalAccuracy_ReturnsExpectedValue()
        {
            // Arrange
            var mtbench = new MTBench(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var optimalAccuracy = mtbench.GetOptimalAccuracy(0.7f);

            // Assert
            Assert.InRange(optimalAccuracy, 0, 10); // MT-Bench typically uses scores from 0 to 10
        }

        [Fact]
        public void GetModelAccuracy_ReturnsExpectedValue()
        {
            // Arrange
            var mtbench = new MTBench(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var strongAccuracy = mtbench.GetModelAccuracy("strong_model");
            var weakAccuracy = mtbench.GetModelAccuracy("weak_model");

            // Assert
            Assert.InRange(strongAccuracy, 0, 10); // MT-Bench typically uses scores from 0 to 10
            Assert.InRange(weakAccuracy, 0, 10);
        }
    }
}