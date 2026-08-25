#![forbid(unsafe_code)]

use crate::learner::SleepLearner;
use crate::memory::EpisodeMemory;
use crate::model::CorticalModel;
use crate::Result;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use candle_core::{IndexOp, Tensor};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCallPayload,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCallPayload {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CorticalMetrics {
    pub cycles: usize,
    pub surprise: f32,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceDelta {
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cortical_metrics: Option<CorticalMetrics>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceDeltaWrapper {
    pub index: usize,
    pub delta: ChatChoiceDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDeltaWrapper>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceResponse {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cortical_metrics: Option<CorticalMetrics>,
}

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub episode_id: i64,
    pub feedback: i32, // +1: positive, -1: negative
}

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<CorticalModel>,
    pub tokenizer: Arc<Tokenizer>,
    pub memory: Arc<Mutex<EpisodeMemory>>,
    pub learner: Arc<SleepLearner>,
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/cotier/metrics", get(metrics_handler))
        .route("/v1/cotier/feedback", post(feedback_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn format_messages_to_prompt(
    messages: &[ChatMessage],
    tools: &Option<Vec<ToolDefinition>>,
) -> String {
    let mut prompt = String::new();

    if let Some(tool_list) = tools {
        if !tool_list.is_empty() {
            let tools_json = serde_json::to_string(tool_list).unwrap_or_default();
            prompt.push_str(&format!(
                "<|im_start|>system\nTools: {}<|im_end|>\n",
                tools_json
            ));
        }
    }

    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role,
            msg.content.as_deref().unwrap_or_default()
        ));
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn extract_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    if let Some(start) = text.find("<tool_call>") {
        if let Some(end) = text.find("</tool_call>") {
            let json_str = &text[start + 11..end].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = val["name"].as_str().unwrap_or("unknown").to_string();
                let args = val["arguments"].to_string();
                return Some(vec![ToolCall {
                    id: format!(
                        "call_{}",
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis()
                    ),
                    r#type: "function".to_string(),
                    function: FunctionCallPayload {
                        name,
                        arguments: args,
                    },
                }]);
            }
        }
    }
    None
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    state.learner.record_activity();

    let is_stream = req.stream.unwrap_or(false);
    let prompt_text = format_messages_to_prompt(&req.messages, &req.tools);

    let encoding = match state.tokenizer.encode(prompt_text.clone(), true) {
        Ok(enc) => enc,
        Err(e) => {
            error!("Failed to encode prompt: {:?}", e);
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let max_tokens = req.max_tokens.unwrap_or(256);

    if is_stream {
        let stream = generate_sse_stream(state, req.model, prompt_text, prompt_ids, max_tokens);
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match generate_non_streaming(&state, &req.model, &prompt_text, &prompt_ids, max_tokens)
            .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => {
                error!("Generation error: {:?}", e);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    }
}

async fn generate_non_streaming(
    state: &AppState,
    model_name: &str,
    prompt_text: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<ChatCompletionResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let id = format!("chatcmpl-{}", now);

    let mut generated_ids = Vec::new();
    let mut current_ids = prompt_ids.to_vec();
    let max_cycles = 1;
    let mut avg_surprise = 0.0f32;

    let eos_id = state.tokenizer.token_to_id("<|im_end|>");

    for _ in 0..max_tokens {
        let input_tensor = Tensor::from_vec(
            current_ids.clone(),
            (1, current_ids.len()),
            &state.model.device,
        )?;
        let logits = state.model.forward(&input_tensor)?;
        let last_logits = logits.i((0, current_ids.len() - 1, ..))?;
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        // Compute softmax surprise: -log(p)
        let probs = candle_nn::ops::softmax(&last_logits, 0)?;
        let p = probs
            .i(next_token as usize)?
            .to_scalar::<f32>()
            .unwrap_or(0.5);
        let surprise = -p.clamp(1e-7, 1.0).ln();
        avg_surprise = (avg_surprise + surprise) / 2.0;

        if Some(next_token) == eos_id {
            break;
        }

        generated_ids.push(next_token);
        current_ids.push(next_token);
    }

    let response_text = state
        .tokenizer
        .decode(&generated_ids, true)
        .unwrap_or_default();
    let tool_calls = extract_tool_calls(&response_text);

    // Save episode to SQLite in background
    let session_id = format!("sess_{}", now);
    let prompt_saved = prompt_text.to_string();
    let response_saved = response_text.clone();
    let mem_arc = state.memory.clone();
    tokio::spawn(async move {
        let mut mem = mem_arc.lock().await;
        let _ = mem.save_episode(
            &session_id,
            &prompt_saved,
            &response_saved,
            avg_surprise,
            max_cycles,
            0,
        );
    });

    let message = ChatMessage {
        role: "assistant".to_string(),
        content: if tool_calls.is_some() {
            None
        } else {
            Some(response_text)
        },
        tool_calls,
    };

    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model: model_name.to_string(),
        choices: vec![ChatChoiceResponse {
            index: 0,
            message,
            finish_reason: "stop".to_string(),
        }],
        cortical_metrics: Some(CorticalMetrics {
            cycles: max_cycles,
            surprise: avg_surprise,
        }),
    })
}

fn generate_sse_stream(
    state: AppState,
    model_name: String,
    prompt_text: String,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
) -> impl Stream<Item = std::result::Result<Event, Infallible>> {
    async_stream::stream! {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs();
        let chunk_id = format!("chatcmpl-{}", now);
        let mut generated_ids = Vec::new();
        let mut current_ids = prompt_ids;
        let eos_id = state.tokenizer.token_to_id("<|im_end|>");
        let mut total_surprise = 0.0f32;
        let mut steps = 0;

        for _ in 0..max_tokens {
            let input_tensor = match Tensor::from_vec(current_ids.clone(), (1, current_ids.len()), &state.model.device) {
                Ok(t) => t,
                Err(_) => break,
            };

            let logits = match state.model.forward(&input_tensor) {
                Ok(l) => l,
                Err(_) => break,
            };

            let last_logits = match logits.i((0, current_ids.len() - 1, ..)) {
                Ok(l) => l,
                Err(_) => break,
            };

            let next_token = match last_logits.argmax(0).and_then(|t| t.to_scalar::<u32>()) {
                Ok(t) => t,
                Err(_) => break,
            };

            if Some(next_token) == eos_id {
                break;
            }

            steps += 1;
            let probs = candle_nn::ops::softmax(&last_logits, 0).unwrap_or(last_logits.clone());
            let p = probs.i(next_token as usize).and_then(|t| t.to_scalar::<f32>()).unwrap_or(0.5);
            let surprise = -p.clamp(1e-7, 1.0).ln();
            total_surprise += surprise;

            generated_ids.push(next_token);
            current_ids.push(next_token);

            let token_text = state.tokenizer.decode(&[next_token], false).unwrap_or_default();

            let chunk = ChatCompletionChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk",
                created: now,
                model: model_name.clone(),
                choices: vec![ChatChoiceDeltaWrapper {
                    index: 0,
                    delta: ChatChoiceDelta {
                        content: Some(token_text),
                        tool_calls: None,
                        cortical_metrics: Some(CorticalMetrics {
                            cycles: 1,
                            surprise,
                        }),
                    },
                    finish_reason: None,
                }],
            };

            if let Ok(json_str) = serde_json::to_string(&chunk) {
                yield Ok(Event::default().data(json_str));
            }
        }

        // Send final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: chunk_id,
            object: "chat.completion.chunk",
            created: now,
            model: model_name,
            choices: vec![ChatChoiceDeltaWrapper {
                index: 0,
                delta: ChatChoiceDelta {
                    content: None,
                    tool_calls: None,
                    cortical_metrics: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        if let Ok(json_str) = serde_json::to_string(&final_chunk) {
            yield Ok(Event::default().data(json_str));
        }

        yield Ok(Event::default().data("[DONE]"));

        // Save dialogue into SQLite episode memory
        let full_response = state.tokenizer.decode(&generated_ids, true).unwrap_or_default();
        let avg_surprise = if steps > 0 { total_surprise / steps as f32 } else { 0.0 };
        let session_id = format!("sess_{}", now);
        let mem = state.memory.clone();
        tokio::spawn(async move {
            let mut m = mem.lock().await;
            let _ = m.save_episode(&session_id, &prompt_text, &full_response, avg_surprise, 1, 0);
        });
    }
}

async fn models_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "cotier-0.5b",
                "object": "model",
                "created": 1700000000,
                "owned_by": "cotier",
                "permission": []
            }
        ]
    }))
}

async fn metrics_handler(State(state): State<AppState>) -> Response {
    let mem = state.memory.lock().await;
    match mem.get_stats() {
        Ok(stats) => Json(serde_json::json!({
            "memory": stats,
            "idle_seconds": state.learner.get_idle_seconds(),
            "is_preempted": state.learner.is_interrupted(),
        }))
        .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn feedback_handler(
    State(state): State<AppState>,
    Json(req): Json<FeedbackRequest>,
) -> Response {
    let mut mem = state.memory.lock().await;
    match mem.update_feedback(req.episode_id, req.feedback) {
        Ok(()) => {
            info!(
                "Received user feedback {} for episode {}",
                req.feedback, req.episode_id
            );
            Json(serde_json::json!({"status": "success", "episode_id": req.episode_id}))
                .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
