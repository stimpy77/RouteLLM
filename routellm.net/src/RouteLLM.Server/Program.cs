using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RouteLLM.Core;
using RouteLLM.Routers;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Configure RouteLLM Controller
var routeLLMController = new Controller(
    routers: new[] { "random", "bert", "sw_ranking", "mf" },
    strongModel: "gpt-4-1106-preview",
    weakModel: "mistralai/Mixtral-8x7B-Instruct-v0.1"
);
builder.Services.AddSingleton(routeLLMController);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

app.Run();
