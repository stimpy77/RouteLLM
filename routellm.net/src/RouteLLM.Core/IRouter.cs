using System.Threading.Tasks;

namespace RouteLLM.Core
{
    public interface IRouter
    {
        Task<float> CalculateStrongWinRate(string prompt);
        Task<string> Route(string prompt, float threshold, ModelPair routedPair);
    }
}