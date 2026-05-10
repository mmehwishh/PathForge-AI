using Microsoft.AspNetCore.Mvc;
using PathGenerator.Infrastructure;

namespace PathGenerator.Api.Controllers;

[ApiController]
[Route("api/learning-path")]
public sealed class LearningPathController : ControllerBase
{
    private readonly MLServiceClient _mlServiceClient;

    public LearningPathController(MLServiceClient mlServiceClient)
    {
        _mlServiceClient = mlServiceClient;
    }

    [HttpGet("topics")]
    public async Task<ActionResult<TopicsResponse>> Topics(CancellationToken cancellationToken)
    {
        try
        {
            var response = await _mlServiceClient.GetTopicsAsync(cancellationToken);
            return Ok(response);
        }
        catch (MLServiceException ex)
        {
            return StatusCode((int)ex.StatusCode, new
            {
                detail = ex.Message,
                code = ex.Code,
                available_topics = ex.AvailableTopics
            });
        }
        catch (HttpRequestException ex)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new
            {
                detail = $"ML service is unavailable: {ex.Message}"
            });
        }
    }

    [HttpPost("generate")]
    public async Task<ActionResult<LearningPathResponse>> Generate(
        [FromBody] LearningPathRequest request,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.PreferredTopic))
        {
            return BadRequest(new { detail = "Preferred topic is required." });
        }

        if (string.IsNullOrWhiteSpace(request.ExperienceLevel))
        {
            return BadRequest(new { detail = "Experience level is required." });
        }

        if (request.StudyHours <= 0)
        {
            return BadRequest(new { detail = "Study hours must be greater than zero." });
        }

        try
        {
            var response = await _mlServiceClient.GetLearningPathAsync(request, cancellationToken);
            return Ok(response);
        }
        catch (MLServiceException ex)
        {
            return StatusCode((int)ex.StatusCode, new
            {
                detail = ex.Message,
                code = ex.Code,
                available_topics = ex.AvailableTopics
            });
        }
        catch (HttpRequestException ex)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new
            {
                detail = $"ML service is unavailable: {ex.Message}"
            });
        }
    }

    [HttpPost("recommendations")]
    public async Task<ActionResult<RecommendationResponse>> Recommendations(
        [FromBody] RecommendationRequest request,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.PreferredTopic))
        {
            return BadRequest(new { detail = "Preferred topic is required." });
        }

        if (string.IsNullOrWhiteSpace(request.ExperienceLevel))
        {
            return BadRequest(new { detail = "Experience level is required." });
        }

        if (request.StudyHours <= 0)
        {
            return BadRequest(new { detail = "Study hours must be greater than zero." });
        }

        try
        {
            var response = await _mlServiceClient.GetRecommendationsAsync(request, cancellationToken);
            return Ok(response);
        }
        catch (MLServiceException ex)
        {
            return StatusCode((int)ex.StatusCode, new
            {
                detail = ex.Message,
                code = ex.Code,
                available_topics = ex.AvailableTopics
            });
        }
        catch (HttpRequestException ex)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new
            {
                detail = $"ML service is unavailable: {ex.Message}"
            });
        }
    }
}
