using System;
using System.Collections.Generic;
using RouteLLM.Routers;

namespace RouteLLM.Core
{
    public static class RouterFactory
    {
        public static IRouter CreateRouter(string routerName, Dictionary<string, object> config = null)
        {
            config ??= new Dictionary<string, object>();

            return routerName.ToLower() switch
            {
                "random" => new RandomRouter(),
                "causal_llm" => new CausalLLMRouter(
                    checkpointPath: GetConfigValue<string>(config, "checkpoint_path"),
                    scoreThreshold: GetConfigValue<float>(config, "score_threshold", 4),
                    specialTokens: GetConfigValue<string[]>(config, "special_tokens"),
                    numOutputs: GetConfigValue<int>(config, "num_outputs", 5),
                    modelType: GetConfigValue<string>(config, "model_type", "causal"),
                    modelId: GetConfigValue<string>(config, "model_id", "meta-llama/Meta-Llama-3-8B"),
                    flashAttention2: GetConfigValue<bool>(config, "flash_attention_2", false)
                ),
                "bert" => new BERTRouter(
                    checkpointPath: GetConfigValue<string>(config, "checkpoint_path"),
                    numLabels: GetConfigValue<int>(config, "num_labels", 3)
                ),
                "sw_ranking" => new SWRankingRouter(
                    arenaBattleDatasets: GetConfigValue<List<string>>(config, "arena_battle_datasets"),
                    arenaEmbeddingDatasets: GetConfigValue<List<string>>(config, "arena_embedding_datasets"),
                    strongModel: GetConfigValue<string>(config, "strong_model", "gpt-4-1106-preview"),
                    weakModel: GetConfigValue<string>(config, "weak_model", "mixtral-8x7b-instruct-v0.1"),
                    numTiers: GetConfigValue<int>(config, "num_tiers", 10)
                ),
                "mf" => new MatrixFactorizationRouter(
                    checkpointPath: GetConfigValue<string>(config, "checkpoint_path"),
                    strongModel: GetConfigValue<string>(config, "strong_model", "gpt-4-1106-preview"),
                    weakModel: GetConfigValue<string>(config, "weak_model", "mixtral-8x7b-instruct-v0.1"),
                    hiddenSize: GetConfigValue<int>(config, "hidden_size", 128),
                    numModels: GetConfigValue<int>(config, "num_models", 64),
                    textDim: GetConfigValue<int>(config, "text_dim", 1536),
                    numClasses: GetConfigValue<int>(config, "num_classes", 1),
                    useProj: GetConfigValue<bool>(config, "use_proj", true)
                ),
                _ => throw new ArgumentException($"Unsupported router type: {routerName}")
            };
        }

        private static T GetConfigValue<T>(Dictionary<string, object> config, string key, T defaultValue = default)
        {
            if (config.TryGetValue(key, out object value))
            {
                return (T)Convert.ChangeType(value, typeof(T));
            }
            return defaultValue;
        }
    }
}