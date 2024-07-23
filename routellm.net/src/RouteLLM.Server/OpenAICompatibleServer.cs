using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using RouteLLM.Core;

namespace RouteLLM.Server
{
    [ApiController]
    [Route("v1")]
    public class OpenAICompatibleServer : ControllerBase
    {
        private readonly Controller routeLLMController;

        public OpenAICompatibleServer(Controller routeLLMController)
        {
            this.routeLLMController = routeLLMController;
        }

        [HttpPost("chat/completions")]
        public async Task<IActionResult> CreateChatCompletion([FromBody] CompletionRequest request)
        {
            try
            {
                var response = await routeLLMController.Completion(request);
                return Ok(response);
            }
            catch (RoutingError ex)
            {
                return BadRequest(new { error = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = "An unexpected error occurred.", details = ex.Message });
            }
        }

        [HttpPost("completions")]
        public async Task<IActionResult> CreateCompletion([FromBody] CompletionRequest request)
        {
            try
            {
                var response = await routeLLMController.Completion(request);
                return Ok(response);
            }
            catch (RoutingError ex)
            {
                return BadRequest(new { error = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = "An unexpected error occurred.", details = ex.Message });
            }
        }

        [HttpGet("models")]
        public IActionResult ListModels()
        {
            var models = new List<string> { "router-random", "router-bert", "router-sw_ranking", "router-mf" };
            return Ok(new { data = models });
        }

        [HttpGet("models/{model}")]
        public IActionResult RetrieveModel(string model)
        {
            if (!model.StartsWith("router-"))
            {
                return NotFound(new { error = "Model not found" });
            }

            return Ok(new
            {
                id = model,
                object = "model",
                created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                owned_by = "routellm"
            });
        }
    }
}