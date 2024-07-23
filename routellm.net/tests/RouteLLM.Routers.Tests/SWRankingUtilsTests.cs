using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using RouteLLM.Routers;

namespace RouteLLM.Routers.Tests
{
    public class SWRankingUtilsTests
    {
        [Fact]
        public void LoadAndPreprocessArenaDatasets_ReturnsValidData()
        {
            // Arrange
            var datasets = new List<string> { "path/to/test/dataset.json" };
            // You'll need to create a test JSON file with sample data

            // Act
            var result = SWRankingUtils.LoadAndPreprocessArenaDatasets(datasets);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result);
            Assert.All(result, item => Assert.NotEqual("tie", item.Winner));
            Assert.All(result, item => Assert.NotEqual(item.ModelA, item.ModelB));
        }

        [Fact]
        public void LoadArenaEmbeddings_ReturnsValidEmbeddings()
        {
            // Arrange
            var datasets = new List<string> { "path/to/test/embeddings.csv" };
            // You'll need to create a test CSV file with sample embeddings

            // Act
            var result = SWRankingUtils.LoadArenaEmbeddings(datasets);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.GetLength(0) > 0);
            Assert.True(result.GetLength(1) > 0);
        }

        [Fact]
        public void ComputeEloMleWithTie_ReturnsValidRatings()
        {
            // Arrange
            var arenaData = new List<ArenaDataItem>
            {
                new ArenaDataItem { ModelA = "ModelA", ModelB = "ModelB", Winner = "ModelA" },
                new ArenaDataItem { ModelA = "ModelB", ModelB = "ModelC", Winner = "ModelC" },
                new ArenaDataItem { ModelA = "ModelC", ModelB = "ModelA", Winner = "ModelA" }
            };

            // Act
            var result = SWRankingUtils.ComputeEloMleWithTie(arenaData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Count);
            Assert.All(result.Values, rating => Assert.InRange(rating, 1000, 2000));
        }

        [Fact]
        public void ComputeTiers_ReturnsValidTiers()
        {
            // Arrange
            var modelRatings = new Dictionary<string, float>
            {
                { "ModelA", 1800 },
                { "ModelB", 1600 },
                { "ModelC", 1400 },
                { "ModelD", 1200 }
            };

            // Act
            var result = SWRankingUtils.ComputeTiers(modelRatings, 2);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(4, result.Count);
            Assert.Equal(0, result["ModelA"]);
            Assert.Equal(0, result["ModelB"]);
            Assert.Equal(1, result["ModelC"]);
            Assert.Equal(1, result["ModelD"]);
        }

        [Fact]
        public void UpdateArenaDataFrameWithTiers_UpdatesCorrectly()
        {
            // Arrange
            var arenaDF = new List<ArenaDataItem>
            {
                new ArenaDataItem { ModelA = "ModelA", ModelB = "ModelB" },
                new ArenaDataItem { ModelA = "ModelC", ModelB = "ModelD" }
            };
            var model2tier = new Dictionary<string, int>
            {
                { "ModelA", 0 },
                { "ModelB", 1 },
                { "ModelC", 1 },
                { "ModelD", 2 }
            };

            // Act
            SWRankingUtils.UpdateArenaDataFrameWithTiers(arenaDF, model2tier);

            // Assert
            Assert.Equal(0, arenaDF[0].ModelATier);
            Assert.Equal(1, arenaDF[0].ModelBTier);
            Assert.Equal(1, arenaDF[1].ModelATier);
            Assert.Equal(2, arenaDF[1].ModelBTier);
        }
    }
}