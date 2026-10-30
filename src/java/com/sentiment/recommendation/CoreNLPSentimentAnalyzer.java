package com.sentiment.recommendation;

import java.util.Properties;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;

/**
 * Stanford CoreNLP Java-based Sentiment Analysis Pipeline.
 * 
 * As described in Section III-B & Section III-E of the research paper:
 * "Sentiment Analysis, Customer Segmentation and Recommendation System"
 * (Grenze / Scopus ACT-2025).
 * 
 * Performs linguistic parsing, POS tagging, and recursive sentiment analysis,
 * producing continuous emotion scores across 5 sentiment levels:
 * 0: Very Negative, 1: Negative, 2: Neutral, 3: Positive, 4: Very Positive.
 */
public class CoreNLPSentimentAnalyzer {

    private final StanfordCoreNLP pipeline;

    public CoreNLPSentimentAnalyzer() {
        Properties props = new Properties();
        // CoreNLP annotators for linguistic parsing & sentiment analysis
        props.setProperty("annotators", "tokenize, ssplit, pos, parse, sentiment");
        props.setProperty("coref.algorithm", "neural");
        this.pipeline = new StanfordCoreNLP(props);
    }

    /**
     * Analyzes raw customer review text and returns sentiment classification
     * and continuous sentiment intensity score.
     */
    public SentimentResult analyzeText(String customerReview) {
        if (customerReview == null || customerReview.trim().isEmpty()) {
            return new SentimentResult("Neutral", 2.0);
        }

        CoreDocument coreDocument = new CoreDocument(customerReview);
        this.pipeline.annotate(coreDocument);

        double totalScore = 0.0;
        int sentenceCount = 0;
        String predominantSentiment = "Neutral";

        for (CoreSentence sentence : coreDocument.sentences()) {
            String sentiment = sentence.sentiment();
            predominantSentiment = sentiment;
            // Stanford CoreNLP returns sentiment tree: 0 to 4
            int sentimentScore = sentence.sentimentTree().score();
            totalScore += sentimentScore;
            sentenceCount++;
        }

        double averageScore = sentenceCount > 0 ? (totalScore / sentenceCount) : 2.0;
        return new SentimentResult(predominantSentiment, averageScore);
    }

    public static class SentimentResult {
        private final String sentimentClass;
        private final double sentimentScore;

        public SentimentResult(String sentimentClass, double sentimentScore) {
            this.sentimentClass = sentimentClass;
            this.sentimentScore = sentimentScore;
        }

        public String getSentimentClass() {
            return sentimentClass;
        }

        public double getSentimentScore() {
            return sentimentScore;
        }

        @Override
        public String toString() {
            return String.format("Sentiment: %s (Score: %.2f / 4.00)", sentimentClass, sentimentScore);
        }
    }

    public static void main(String[] args) {
        System.out.println("Initializing Stanford CoreNLP Sentiment Pipeline...");
        CoreNLPSentimentAnalyzer analyzer = new CoreNLPSentimentAnalyzer();

        String sampleReview = "The product quality is absolutely outstanding! Fast delivery and great customer support.";
        SentimentResult result = analyzer.analyzeText(sampleReview);
        System.out.println("Customer Review: " + sampleReview);
        System.out.println("Result: " + result);
    }
}
