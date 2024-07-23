using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using RouteLLM.Core;

namespace RouteLLM.Core.Tests
{
    public class EmbeddingUtilsTests
    {
        [Fact]
        public async Task GetEmbedding_ReturnsValidEmbedding()
        {
            // Arrange
            string text = "Hello, world!";

            // Act
            var embedding = await EmbeddingUtils.GetEmbedding(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.True(embedding.Length > 0);
        }

        [Fact]
        public void CosineSimilarity_CalculatesCorrectly()
        {
            // Arrange
            float[] a = new float[] { 1, 2, 3 };
            float[] b = new float[] { 4, 5, 6 };

            // Act
            float similarity = EmbeddingUtils.CosineSimilarity(a, b);

            // Assert
            Assert.InRange(similarity, 0.97, 0.98); // Approximate value
        }

        [Fact]
        public void GetTopKSimilar_ReturnsCorrectResults()
        {
            // Arrange
            float[] query = new float[] { 1, 2, 3 };
            List<float[]> embeddings = new List<float[]>
            {
                new float[] { 1, 2, 3 },
                new float[] { 4, 5, 6 },
                new float[] { 7, 8, 9 },
                new float[] { 2, 3, 4 }
            };

            // Act
            var results = EmbeddingUtils.GetTopKSimilar(query, embeddings, 2);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.Equal(0, results[0].Index);
            Assert.Equal(3, results[1].Index);
        }
    }
}