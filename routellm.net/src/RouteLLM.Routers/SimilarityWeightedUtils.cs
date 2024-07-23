using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace RouteLLM.Routers
{
    public static class SimilarityWeightedUtils
    {
        public static double CosineSimilarity(Vector<float> a, Vector<float> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must have the same dimension");

            double dotProduct = Vector.Dot(a, b);
            double magnitudeA = Math.Sqrt(Vector.Dot(a, a));
            double magnitudeB = Math.Sqrt(Vector.Dot(b, b));

            return dotProduct / (magnitudeA * magnitudeB);
        }

        public static List<(int, double)> GetTopKSimilar(Vector<float> query, List<Vector<float>> embeddings, int k)
        {
            var similarities = embeddings.Select((e, i) => (Index: i, Similarity: CosineSimilarity(query, e)))
                                         .OrderByDescending(x => x.Similarity)
                                         .Take(k)
                                         .ToList();

            return similarities.Select(x => (x.Index, x.Similarity)).ToList();
        }

        // Add more utility functions as needed
    }
}
