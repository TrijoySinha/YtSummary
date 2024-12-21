import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_transcript_section(youtube_video_url, start_time, end_time):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)

        section_text = ""
        for entry in transcript_data:
            if start_time <= entry['start'] <= end_time:
                section_text += " " + entry['text']

        if not section_text:
            return "No transcript found for the given time range."

        return section_text

    except Exception as e:
        return f"Error in extracting transcript: {e}"

def generate_distilbart_summary(transcript_text, max_words):
    try:
        
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        inputs = tokenizer.encode("summarize: " + transcript_text, return_tensors="pt", max_length=1024, truncation=True)

        max_tokens = int(max_words * 1.33)

        summary_ids = model.generate(
            inputs,
            max_length=max_tokens,
            min_length=int(max_tokens * 0.5),  
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        summary_words = summary.split()[:max_words]
        final_summary = " ".join(summary_words)
        return final_summary

    except Exception as e:
        return f"Error in generating summary: {e}"

def search_in_summary(summary, search_term):
    try:
        if search_term.lower() in summary.lower():
            return f"The term '{search_term}' was found in the summary."
        else:
            return f"The term '{search_term}' was not found in the summary."
    except Exception as e:
        return f"Error in searching the summary: {e}"

st.title("ShortIt")
youtube_link = st.text_input("Enter YouTube Video Link:")

start_time = st.number_input("Enter Start Time (in seconds):", min_value=0, step=1, value=0)
end_time = st.number_input("Enter End Time (in seconds):", min_value=1, step=1, value=60)

summary_length = st.slider("Select Summary Length (in words):", min_value=50, max_value=500, value=250, step=10)

search_term = st.text_input("Enter a word or phrase to search in the summary:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Summary"):
    transcript_section = extract_transcript_section(youtube_link, start_time, end_time)

    if "Error" in transcript_section or "No transcript" in transcript_section:
        st.error(transcript_section)
    else:
        summary = generate_distilbart_summary(transcript_section, max_words=summary_length)

        st.markdown("## Section Summary:")
        st.write(summary)

        if search_term:
            search_result = search_in_summary(summary, search_term)
            st.markdown("## Search Result:")
            st.write(search_result)