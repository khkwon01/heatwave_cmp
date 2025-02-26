USE ecommerce;

DROP PROCEDURE IF EXISTS SENTIMENT_ANALYSIS;
CREATE PROCEDURE SENTIMENT_ANALYSIS(
  IN review TEXT,
  IN review_id INT
) LANGUAGE JAVASCRIPT AS $$

  let prompt = `review에 내용을 긍정 또는 부정으로 분류해 주세요.\n${review}. 
    분류 결과는 "긍정" 또는 "부정"으로 한단어로 표현해 주세요.\n반응:`;
 
  let sentiment = ml.generate(prompt, {model_id: "llama3-8b-instruct-v1"});
  let processed_sentiment = sentiment.toUpperCase().search("긍정") ? "긍정" : "부정";

  let sql = session.prepare(`UPDATE reviews SET sentiment = ? WHERE id = ?`);
  sql.bind(processed_sentiment, review_id).execute();
$$;

DROP PROCEDURE IF EXISTS SUMMARIZE_TRANSLATE;
CREATE PROCEDURE SUMMARIZE_TRANSLATE(
  IN product_id INT,
  IN sentiment VARCHAR(20),
  IN language VARCHAR(64),
  OUT processed_summary TEXT
) LANGUAGE JAVASCRIPT AS $$

  let sql = session.prepare("SELECT review_text FROM reviews WHERE product_id = ? AND sentiment = ?");
  let reviews = sql.bind(product_id, sentiment).execute();
  let all_reviews = Array.from(reviews).map(review => review.review_text).join("\n");

  let summary = ml.generate(all_reviews, {model_id: "llama3-8b-instruct-v1", task: "summarization", language: "ko"});
  processed_summary = summary.trim();

  if (language != "ko") {  
    let prompt = `Translate the Original Text to ${language}. \n 
                           - Original Text: "${processed_summary}"\n - ${language} Translation:`;
    let translation = ml.generate(prompt, {max_tokens: 800, language: language}); 
    processed_summary = translation.split('\n')[0];
  }
$$;
