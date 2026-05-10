using PathGenerator.Infrastructure;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

builder.Services.AddCors(options =>
{
    options.AddPolicy("Frontend", policy =>
    {
        policy
            .WithOrigins("http://localhost:5173", "http://127.0.0.1:5173")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

builder.Services.AddHttpClient<MLServiceClient>(client =>
{
    var baseUrl = builder.Configuration["MLService:BaseUrl"] ?? "http://127.0.0.1:8000/";
    client.BaseAddress = new Uri(baseUrl);
});

var app = builder.Build();

app.UseCors("Frontend");
app.MapControllers();

app.MapGet("/health", () => Results.Ok(new
{
    status = "healthy",
    service = "path_generator_api"
}));

app.Run();
