using System;
using System.Threading.Tasks;
using RouteLLM.Core;

namespace RouteLLM.Routers
{
    public class RandomRouter : IRouter
    {
        private readonly Random _random = new Random();

        public Task<float> CalculateStrongWinRate(string prompt)
        {
            return Task.FromResult((float)_random.NextDouble());
        }

        public async Task<string> Route(string prompt, float threshold, ModelPair routedPair)
        {
            float winRate = await CalculateStrongWinRate(prompt);
            return winRate >= threshold ? routedPair.Strong : routedPair.Weak;
        }
    }
}
