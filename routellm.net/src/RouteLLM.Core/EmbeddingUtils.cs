using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RouteLLM.Core
{
    public static class EmbeddingUtils
    {
        private static readonly HttpClient httpClient = new HttpClient();
        private static readonly string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        private static readonly string apiBase = Environment.GetEnvironmentVariable("OPENAI_API_BASE") ?? "https://api.openai.com/v1";

        public static async Task<float[]> GetEmbedding(string text, string model = "text-embedding-3-small")
        {
            var request = new HttpRequestMessage(HttpMethod.Post, $"{apiBase}/embeddings");
            request.Headers.Add("Authorization", $"Bearer {apiKey}");

            var requestBody = new
            {
                input = text,
                model = model
            };

            var json = JsonSerializer.Serialize(requestBody);
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseBody = await response.Content.ReadAsStringAsync();
            var embeddingResponse = JsonSerializer.Deserialize<EmbeddingResponse>(responseBody);

            return embeddingResponse.Data[0].Embedding;
        }

        public static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must have the same dimension");

            float dotProduct = 0;
            float normA = 0;
            float normB = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            return dotProduct / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB));
        }

        public static List<(int Index, float Similarity)> GetTopKSimilar(float[] query, List<float[]> embeddings, int k)
        {
            var similarities = new List<(int Index, float Similarity)>();

            for (int i = 0; i < embeddings.Count; i++)
            {
                float similarity = CosineSimilarity(query, embeddings[i]);
                similarities.Add((i, similarity));
            }

            return similarities.OrderByDescending(x => x.Similarity).Take(k).ToList();
        }

        private class EmbeddingResponse
        {
            public List<EmbeddingData> Data { get; set; }
        }

        private class EmbeddingData
        {
            public float[] Embedding { get; set; }
        }
    }
}