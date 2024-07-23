using System;

namespace RouteLLM.Core
{
    public class RoutingError : Exception
    {
        public RoutingError(string message) : base(message) { }
    }
}