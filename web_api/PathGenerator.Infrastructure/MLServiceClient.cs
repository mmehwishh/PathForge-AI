using System.Net.Http.Json;

namespace PathGenerator.Infrastructure {
    public class MLServiceClient {
        private readonly HttpClient _http;
        public MLServiceClient(HttpClient http) { _http = http; }

        public async Task<object> GetLearningPath(object userPrefs) {
            var response = await _http.PostAsJsonAsync("http://localhost:8000/predict", userPrefs);
            return await response.Content.ReadFromJsonAsync<object>();
        }
    }
}