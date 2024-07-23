using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using Moq;
using RouteLLM.Core;
using RouteLLM.Evaluations;

namespace RouteLLM.Evaluations.Tests
{
    public class GSM8KTests
    {
        [Fact]
        public void Evaluate_ReturnsExpectedResults()
        {
            // Arrange
            var mockController = new Mock<Controller>();
            mockController.Setup(c => c.BatchCalculateWinRate(It.IsAny<List<string>>(), It.IsAny<string>()))
                .ReturnsAsync(new List<float> { 0.7f, 0.8f, 0.6f, 0.9f, 0.5f });

            var gsm8k = new GSM8K(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var results = gsm8k.Evaluate(mockController.Object, "test_router", 3, false).ToList();

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);
            Assert.All(results, r =>
            {
                Assert.InRange(r.Threshold, 0, 1);
                Assert.InRange(r.Accuracy, 0, 100);
                Assert.NotNull(r.ModelCounts);
                Assert.Equal(5, r.Total);
            });
        }

        [Fact]
        public void GetOptimalAccuracy_ReturnsExpectedValue()
        {
            // Arrange
            var gsm8k = new GSM8K(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var optimalAccuracy = gsm8k.GetOptimalAccuracy(0.7f);

            // Assert
            Assert.InRange(optimalAccuracy, 0, 100);
        }

        [Fact]
        public void GetModelAccuracy_ReturnsExpectedValue()
        {
            // Arrange
            var gsm8k = new GSM8K(new ModelPair("strong_model", "weak_model"), false);

            // Act
            var strongAccuracy = gsm8k.GetModelAccuracy("strong_model");
            var weakAccuracy = gsm8k.GetModelAccuracy("weak_model");

            // Assert
            Assert.InRange(strongAccuracy, 0, 100);
            Assert.InRange(weakAccuracy, 0, 100);
        }
    }
}