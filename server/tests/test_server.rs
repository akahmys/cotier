use axum::body::Body;
use axum::http::{Request, StatusCode};
use cotier_server::learner::SleepLearner;
use cotier_server::memory::EpisodeMemory;
use cotier_server::model::CorticalModel;
use cotier_server::server::{create_router, AppState};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::util::ServiceExt;

#[tokio::test]
async fn test_server_routes_and_chat_completion() -> cotier_server::Result<()> {
    let model_dir = PathBuf::from("../models/cotier-0.5b");
    if !model_dir.join("model.safetensors").exists() {
        return Ok(());
    }

    let device = candle_core::Device::Cpu;
    let model = CorticalModel::load(&model_dir, &device)?;
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| cotier_server::CotierError::ModelLoad(e.to_string()))?;

    let tmp_db = std::env::temp_dir().join(format!(
        "cotier_test_{}.sqlite",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let memory = EpisodeMemory::open(&tmp_db)?;

    let state = AppState {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
        memory: Arc::new(Mutex::new(memory)),
        learner: Arc::new(SleepLearner::new()),
    };

    let app = create_router(state);

    // 1. Test GET /v1/models
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .method("GET")
                .body(Body::empty())
                .unwrap_or_default(),
        )
        .await
        .unwrap_or_else(|_| panic!("Route failed"));

    assert_eq!(res.status(), StatusCode::OK);

    // 2. Test GET /v1/cotier/metrics
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/cotier/metrics")
                .method("GET")
                .body(Body::empty())
                .unwrap_or_default(),
        )
        .await
        .unwrap_or_else(|_| panic!("Route failed"));

    assert_eq!(res.status(), StatusCode::OK);

    // 3. Test POST /v1/chat/completions (Non-streaming)
    let chat_req_json = serde_json::json!({
        "model": "cotier-0.5b",
        "messages": [
            {"role": "user", "content": "Hello, Cotier!"}
        ],
        "max_tokens": 5,
        "stream": false
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&chat_req_json)?))
                .unwrap_or_default(),
        )
        .await
        .unwrap_or_else(|_| panic!("Route failed"));

    assert_eq!(res.status(), StatusCode::OK);

    // 4. Test POST /v1/chat/completions (SSE Streaming)
    let stream_req_json = serde_json::json!({
        "model": "cotier-0.5b",
        "messages": [
            {"role": "user", "content": "Tell me a story."}
        ],
        "max_tokens": 5,
        "stream": true
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&stream_req_json)?))
                .unwrap_or_default(),
        )
        .await
        .unwrap_or_else(|_| panic!("Route failed"));

    assert_eq!(res.status(), StatusCode::OK);

    // 5. Test POST /v1/cotier/feedback
    let fb_req = serde_json::json!({
        "episode_id": 1,
        "feedback": 1
    });

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/cotier/feedback")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&fb_req)?))
                .unwrap_or_default(),
        )
        .await
        .unwrap_or_else(|_| panic!("Route failed"));

    assert_eq!(res.status(), StatusCode::OK);

    Ok(())
}
