using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace PathGenerator.Infrastructure;

public sealed class MLServiceClient
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private readonly HttpClient _http;

    public MLServiceClient(HttpClient http)
    {
        _http = http;

        if (_http.BaseAddress is null)
        {
            _http.BaseAddress = new Uri("http://127.0.0.1:8000/");
        }
    }

    public async Task<LearningPathResponse> GetLearningPathAsync(
        LearningPathRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        using var response = await _http.PostAsJsonAsync(
            "predict",
            request,
            JsonOptions,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var error = await ReadErrorDetailAsync(response, cancellationToken);
            throw new MLServiceException(response.StatusCode, error.Message, error.Code, error.AvailableTopics);
        }

        var payload = await response.Content.ReadFromJsonAsync<LearningPathResponse>(
            JsonOptions,
            cancellationToken);

        return payload ?? throw new MLServiceException(
            response.StatusCode,
            "ML service returned an empty response.");
    }

    public async Task<RecommendationResponse> GetRecommendationsAsync(
        RecommendationRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        using var response = await _http.PostAsJsonAsync(
            "recommendations",
            request,
            JsonOptions,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var error = await ReadErrorDetailAsync(response, cancellationToken);
            throw new MLServiceException(response.StatusCode, error.Message, error.Code, error.AvailableTopics);
        }

        var payload = await response.Content.ReadFromJsonAsync<RecommendationResponse>(
            JsonOptions,
            cancellationToken);

        return payload ?? throw new MLServiceException(
            response.StatusCode,
            "ML service returned an empty response.");
    }

    private static async Task<MLServiceErrorDetails> ReadErrorDetailAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (string.IsNullOrWhiteSpace(body))
        {
            return new MLServiceErrorDetails
            {
                Message = $"ML service failed with status {(int)response.StatusCode} {response.ReasonPhrase}."
            };
        }

        try
        {
            var error = JsonSerializer.Deserialize<FastApiError>(body, JsonOptions);
            return error?.ToErrorDetails() ?? new MLServiceErrorDetails { Message = body };
        }
        catch (JsonException)
        {
            return new MLServiceErrorDetails { Message = body };
        }
    }
}

public class LearningPathRequest
{
    [JsonPropertyName("preferred_topic")]
    public string PreferredTopic { get; set; } = string.Empty;

    [JsonPropertyName("experience_level")]
    public string ExperienceLevel { get; set; } = string.Empty;

    [JsonPropertyName("study_hours")]
    public int StudyHours { get; set; }
}

public sealed class RecommendationRequest : LearningPathRequest
{
    [JsonPropertyName("top_k")]
    public int TopK { get; set; } = 5;
}

public sealed class LearningPathResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("learning_path")]
    public LearningPathDto LearningPath { get; set; } = new();
}

public sealed class RecommendationResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("recommendations")]
    public List<RecommendationDto> Recommendations { get; set; } = new();
}

public sealed class LearningPathDto
{
    [JsonPropertyName("topic")]
    public string Topic { get; set; } = string.Empty;

    [JsonPropertyName("level")]
    public string Level { get; set; } = string.Empty;

    [JsonPropertyName("total_weeks")]
    public int TotalWeeks { get; set; }

    [JsonPropertyName("weekly_hours")]
    public int WeeklyHours { get; set; }

    [JsonPropertyName("phases")]
    public List<string> Phases { get; set; } = new();

    [JsonPropertyName("courses")]
    public List<CourseStepDto> Courses { get; set; } = new();

    [JsonPropertyName("total_estimated_hours")]
    public double TotalEstimatedHours { get; set; }
}

public sealed class CourseStepDto
{
    [JsonPropertyName("week")]
    public int Week { get; set; }

    [JsonPropertyName("start_week")]
    public int StartWeek { get; set; }

    [JsonPropertyName("end_week")]
    public int EndWeek { get; set; }

    [JsonPropertyName("phase")]
    public string Phase { get; set; } = string.Empty;

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("estimated_hours")]
    public double EstimatedHours { get; set; }

    [JsonPropertyName("estimated_weeks")]
    public int EstimatedWeeks { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
}

public sealed class RecommendationDto
{
    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("subject")]
    public string Subject { get; set; } = string.Empty;

    [JsonPropertyName("difficulty")]
    public string Difficulty { get; set; } = string.Empty;

    [JsonPropertyName("rating")]
    public double Rating { get; set; }

    [JsonPropertyName("similarity_score")]
    public double SimilarityScore { get; set; }

    [JsonPropertyName("stage")]
    public string Stage { get; set; } = string.Empty;

    [JsonPropertyName("sequence_rank")]
    public int SequenceRank { get; set; }

    [JsonPropertyName("estimated_total_hours")]
    public double EstimatedTotalHours { get; set; }

    [JsonPropertyName("estimated_weekly_hours")]
    public double EstimatedWeeklyHours { get; set; }

    [JsonPropertyName("match_score")]
    public double MatchScore { get; set; }
}

public sealed class MLServiceException : Exception
{
    public MLServiceException(
        HttpStatusCode statusCode,
        string message,
        string? code = null,
        List<string>? availableTopics = null)
        : base(message)
    {
        StatusCode = statusCode;
        Code = code;
        AvailableTopics = availableTopics ?? new List<string>();
    }

    public HttpStatusCode StatusCode { get; }

    public string? Code { get; }

    public List<string> AvailableTopics { get; }
}

internal sealed class FastApiError
{
    [JsonPropertyName("detail")]
    public JsonElement Detail { get; set; }

    public MLServiceErrorDetails ToErrorDetails()
    {
        if (Detail.ValueKind == JsonValueKind.String)
        {
            return new MLServiceErrorDetails
            {
                Message = Detail.GetString() ?? "ML service request failed."
            };
        }

        if (Detail.ValueKind != JsonValueKind.Object)
        {
            return new MLServiceErrorDetails { Message = Detail.GetRawText() };
        }

        var details = new MLServiceErrorDetails();

        if (Detail.TryGetProperty("code", out var code) &&
            code.ValueKind == JsonValueKind.String)
        {
            details.Code = code.GetString();
        }

        if (Detail.TryGetProperty("message", out var message) &&
            message.ValueKind == JsonValueKind.String)
        {
            details.Message = message.GetString() ?? details.Message;
        }

        if (Detail.TryGetProperty("available_topics", out var topics) &&
            topics.ValueKind == JsonValueKind.Array)
        {
            foreach (var topic in topics.EnumerateArray())
            {
                if (topic.ValueKind == JsonValueKind.String)
                {
                    details.AvailableTopics.Add(topic.GetString()!);
                }
            }
        }

        return details;
    }
}

internal sealed class MLServiceErrorDetails
{
    public string Message { get; set; } = "ML service request failed.";

    public string? Code { get; set; }

    public List<string> AvailableTopics { get; } = new();
}
