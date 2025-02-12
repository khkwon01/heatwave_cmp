USE ecommerce;

SELECT '################### 고객 review 반응 분류 ###################' AS '';

SET @review='Review:이 티셔츠는 편안할 뿐만 아니라 지속 가능한 소재로 만들어졌기 때문에 환상적입니다. 유기농 면은 환경 친화적이면서도 일반 면과 같은 느낌이 듭니다. 하지만 세탁 지침을 따르기 어렵습니다.';
SET @review_id = 1337;

SELECT @review AS "";

INSERT INTO reviews(id, language_code, product_id, customer_id, rating, review_text) VALUES (@review_id,'ko',1,20,4,@review);

SELECT review_text INTO @review_text FROM reviews WHERE id = @review_id;
CALL SENTIMENT_ANALYSIS(@review_text, @review_id);

SELECT @review AS "--- New review ---";
SELECT sentiment AS "--- Sentiment of the new review ---" FROM reviews WHERE id = @review_id;


SELECT '################### 제품에 대한 고객에 review 내용들을 요약 ###################' AS '';

CALL SUMMARIZE_TRANSLATE(1, "긍정", "ko", @positive_korea_summary);
SELECT @positive_korea_summary AS "--- 티셔츠에 대한 긍정적인 리뷰들에 대한 업데이트된 내용 요약 ---";

CALL SUMMARIZE_TRANSLATE(1, "부정", "ko", @negative_korea_summary);
SELECT @negative_korea_summary AS "--- 티셔츠에 대한 긍정적인 리뷰들에 대한 업데이트된 내용 요약 ---";


SELECT '################### 제품에 대한 고객에 review 내용들을 다른 언어로 번역 ###################' AS '';

CALL SUMMARIZE_TRANSLATE(1, "긍정", "en", @positive_english_summary);
SELECT @positive_english_summary AS "--- English summary of positive reviews on the T-Shirt ---";

CALL SUMMARIZE_TRANSLATE(1, "부정", "en", @negative_english_summary);
SELECT @negative_english_summary AS "--- English summary of negative reviews on the T-Shirt ---";
