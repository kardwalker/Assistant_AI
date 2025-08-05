
import os
import openai
import logging
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

import numpy as np
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_summarizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration - Load from .env file
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT") 
AZURE_API_VERSION = "2024-02-01"

# Whisper API Configuration from .env
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")  # Azure Whisper API key
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT")  # Azure Whisper endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI Whisper API (if available)

class AudioSummarizer:
    def __init__(self, use_azure=True, use_whisper_api=True):
        """
        Initialize AudioSummarizer with multiple transcription options
        
        Args:
            use_azure (bool): Use Azure for summarization
            use_whisper_api (bool): Use Azure Whisper API for transcription (faster)
        """
        logger.info("Initializing AudioSummarizer...")
        
        # Configure transcription method - Azure Whisper API preferred
        self.use_whisper_api = use_whisper_api and bool(WHISPER_API_KEY and WHISPER_ENDPOINT)
        if self.use_whisper_api:
            logger.info("✅ Azure Whisper API available for fast transcription")
        else:
            logger.info("Using local Whisper model for transcription")
        
        # Azure is required for summarization
        if not AZURE_API_KEY or not AZURE_ENDPOINT:
            logger.warning("Azure API credentials not found. Summarization will be limited.")
            self.use_azure = False
        else:
            self.use_azure = use_azure
            logger.info("Azure API credentials found and configured")
        
        # Initialize local transcription model (fallback or primary)
        if not self.use_whisper_api:
            try:
                logger.info("Loading local Whisper model for transcription...")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                logger.info("✅ Local Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local Whisper model: {e}")
                if not self.use_whisper_api:
                    raise
        else:
            self.processor = None
            self.model = None
            logger.info("Skipping local Whisper model (using API)")
        
        # Configure Azure OpenAI for summarization
        if self.use_azure:
            try:
                openai.api_type = "azure"
                openai.api_key = AZURE_API_KEY
                openai.api_base = AZURE_ENDPOINT
                openai.api_version = AZURE_API_VERSION
                logger.info("✅ Azure OpenAI configured for summarization")
            except Exception as e:
                logger.error(f"Failed to configure Azure OpenAI: {e}")
                self.use_azure = False
    
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text using Whisper API or local model
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        logger.info(f"Starting transcription for: {os.path.basename(audio_file_path)}")
        
        if not os.path.exists(audio_file_path):
            error_msg = f"Audio file not found: {audio_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            if self.use_whisper_api:
                logger.info("Using Whisper API for fast transcription...")
                return self._transcribe_with_whisper_api(audio_file_path)
            else:
                logger.info("Using local Whisper model for transcription...")
                return self._transcribe_with_huggingface(audio_file_path)
        except Exception as e:
            logger.error(f"Transcription failed with primary method: {e}")
            # Fallback to alternate method
            try:
                if self.use_whisper_api:
                    logger.info("Falling back to local Whisper model...")
                    return self._transcribe_with_huggingface(audio_file_path)
                else:
                    logger.info("Falling back to Whisper API...")
                    return self._transcribe_with_whisper_api(audio_file_path)
            except Exception as e2:
                logger.error(f"Both transcription methods failed: {e2}")
                return ""
    
    def _transcribe_with_whisper_api(self, audio_file_path: str) -> str:
        """Transcribe using Azure Whisper API (fast and accurate)"""
        try:
            logger.info("Using Azure Whisper API for transcription...")
            
            # Azure Whisper API configuration
            import requests
            
            # Determine content type based on file extension
            file_ext = audio_file_path.lower().split('.')[-1]
            if file_ext == 'mp3':
                content_type = 'audio/mpeg'
            elif file_ext == 'wav':
                content_type = 'audio/wav'
            elif file_ext == 'm4a':
                content_type = 'audio/mp4'
            else:
                content_type = 'audio/wav'  # default
            
            # Read the audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            logger.info(f"Audio file size: {len(audio_data)} bytes, Content-Type: {content_type}")
            
            # Set up headers for Azure Whisper API
            headers = {
                'Ocp-Apim-Subscription-Key': WHISPER_API_KEY,
                'Content-Type': content_type
            }
            
            # Azure Speech-to-Text API endpoint (updated)
            url = f"{WHISPER_ENDPOINT.rstrip('/')}/speechtotext/v3.1/transcriptions"
            
            logger.info(f"Calling Azure Speech API: {url}")
            
            # Make the API call with form data (multipart)
            files = {
                'audio': (audio_file_path.split('\\')[-1], audio_data, content_type)
            }
            
            # Use multipart form data instead of raw data
            headers_multipart = {
                'Ocp-Apim-Subscription-Key': WHISPER_API_KEY
            }
            
            response = requests.post(
                url,
                headers=headers_multipart,
                files=files,
                data={
                    'language': 'en-US',
                    'format': 'simple'
                }
            )
            
            logger.info(f"Azure API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('DisplayText', '') or result.get('text', '') or result.get('combinedResults', [{}])[0].get('display', '')
                logger.info(f"✅ Azure Whisper API transcription completed: {len(transcription)} characters")
                return transcription
            else:
                error_msg = f"Azure Whisper API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return ""
            
        except Exception as e:
            logger.error(f"Azure Whisper API transcription error: {e}")
            return ""
    
    def _transcribe_with_huggingface(self, audio_file_path: str) -> str:
        """Transcribe using local Hugging Face Whisper model"""
        try:
            if not LIBROSA_AVAILABLE:
                error_msg = "Librosa not available. Install with: pip install librosa"
                logger.error(error_msg)
                return error_msg
            
            logger.info("Loading audio file...")
            # Load audio file
            audio, sample_rate = librosa.load(audio_file_path, sr=16000)
            logger.info(f"Audio loaded: duration={len(audio)/16000:.2f}s, sample_rate={sample_rate}")
            
            # Process audio in chunks if it's too long
            chunk_length = 30 * 16000  # 30 seconds
            transcription = ""
            total_chunks = (len(audio) + chunk_length - 1) // chunk_length
            
            logger.info(f"Processing {total_chunks} audio chunks...")
            
            for i in range(0, len(audio), chunk_length):
                chunk_num = i // chunk_length + 1
                chunk = audio[i:i + chunk_length]
                
                logger.debug(f"Processing chunk {chunk_num}/{total_chunks}")
                
                # Process with Whisper
                inputs = self.processor(chunk, 
                                      sampling_rate=16000, 
                                      return_tensors="pt")
                
                predicted_ids = self.model.generate(inputs.input_features)
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcription += text + " "
                
                logger.info(f"✅ Processed chunk {chunk_num}/{total_chunks}")
            
            final_transcription = transcription.strip()
            logger.info(f"Transcription completed: {len(final_transcription)} characters")
            return final_transcription
            
        except Exception as e:
            logger.error(f"Local transcription error: {e}")
            return ""
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize the transcribed text using Azure OpenAI (REQUIRED)
        
        Args:
            text (str): Text to summarize
            
        Returns:
            str: Summary or error message
        """
        logger.info("Starting text summarization...")
        
        if not text.strip():
            logger.warning("No text provided for summarization")
            return "No text to summarize."
        
        if not self.use_azure:
            error_msg = "Azure API required for summarization. Please configure AZURE_API_KEY and AZURE_ENDPOINT."
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"Summarizing text: {len(text)} characters")
        return self._summarize_with_azure(text)
    
    def _summarize_with_azure(self, text: str) -> str:
        """Summarize using Azure OpenAI (ONLY APPROACH)"""
        try:
            logger.info("Calling Azure OpenAI for summarization...")
            
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",  # or your deployed model name
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that creates concise and informative summaries of audio transcriptions."
                    },
                    {
                        "role": "user", 
                        "content": f"Create a detailed summary of the following audio transcription in 3-5 sentences. Include key topics, main points, and important details:\n\n{text}"
                    }
                ],
                max_tokens=520,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            logger.info(f"✅ Summarization completed: {len(summary)} characters")
            return summary
            
        except Exception as e:
            error_msg = f"Azure summarization failed: {e}"
            logger.error(error_msg)
            return f"Summarization failed: {str(e)}"
    
    def process_audio_file(self, audio_file_path: str) -> dict:
        """
        Complete audio processing: transcription + summarization
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            dict: Results with transcription and summary
        """
        logger.info(f"🎵 Processing audio file: {os.path.basename(audio_file_path)}")
        
        # Transcribe
        logger.info("📝 Starting audio transcription...")
        transcription = self.transcribe_audio(audio_file_path)
        
        if not transcription:
            logger.error("❌ Transcription failed or returned empty")
            return {
                'audio_file': audio_file_path,
                'transcription': '',
                'summary': 'Failed to transcribe audio.',
                'success': False
            }
        
        logger.info(f"✅ Transcription completed: {len(transcription)} characters")
        
        # Summarize
        logger.info("📄 Starting text summarization...")
        summary = self.summarize_text(transcription)
        
        success = bool(transcription and summary and not summary.startswith("Summarization failed"))
        
        logger.info(f"{'✅' if success else '❌'} Audio processing {'completed' if success else 'failed'}")
        
        return {
            'audio_file': audio_file_path,
            'transcription': transcription,
            'summary': summary,
            'success': success,
            'transcription_method': 'azure_whisper_api' if self.use_whisper_api else 'local_whisper',
            'summarization_method': 'azure_openai' if self.use_azure else 'none'
        }

def process_video_and_summarize(video_file_path: str, use_azure: bool = True) -> dict:
    """
    Complete pipeline: Extract audio from video and create summary
    
    Args:
        video_file_path (str): Path to video file
        use_azure (bool): Whether to use Azure APIs for summarization
        
    Returns:
        dict: Complete results
    """
    # Import here to avoid circular imports
    try:
        from separator import extract_audio_only
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from separator import extract_audio_only
    
    logger.info("🎬 Starting video processing and summarization...")
    
    # Step 1: Extract audio
    logger.info("🔊 Extracting audio from video...")
    audio_file_path = extract_audio_only(video_file_path)
    
    if not audio_file_path:
        return {
            'video_file': video_file_path,
            'audio_file': None,
            'transcription': '',
            'summary': 'Failed to extract audio from video.',
            'success': False
        }
    
    # Step 2: Process audio with optimal transcription method
    # Try Azure Whisper API first (faster), fallback to local model
    use_whisper_api = bool(WHISPER_API_KEY and WHISPER_ENDPOINT)
    logger.info(f"🎵 Using {'Azure Whisper API' if use_whisper_api else 'Local Whisper'} for transcription")
    
    summarizer = AudioSummarizer(use_azure=use_azure, use_whisper_api=use_whisper_api)
    result = summarizer.process_audio_file(audio_file_path)
    
    # Add video file info to result
    result['video_file'] = video_file_path
    result['audio_file'] = audio_file_path
    
    return result

# Example usage
if __name__ == "__main__":
    # Example with video file
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    logger.info("🚀 Starting audio summarization test...")
    
    # Process video and create summary
    result = process_video_and_summarize(video_path, use_azure=True)
    
    if result['success']:
        print("\n" + "="*60)
        print("📋 AUDIO PROCESSING RESULTS")
        print("="*60)
        print(f"🎬 Video: {os.path.basename(result['video_file'])}")
        print(f"🔊 Audio: {os.path.basename(result['audio_file'])}")
        print(f"🎵 Transcription Method: {result.get('transcription_method', 'unknown')}")
        print(f"📄 Summarization Method: {result.get('summarization_method', 'unknown')}")
        print(f"\n📝 Transcription ({len(result['transcription'])} chars):")
        print("-" * 30)
        print(result['transcription'][:500] + "..." if len(result['transcription']) > 500 else result['transcription'])
        print(f"\n📄 Summary:")
        print("-" * 30)
        print(result['summary'])
        print("="*60)
        logger.info("✅ Audio summarization completed successfully")
    else:
        print(f"❌ Processing failed: {result['summary']}")
        logger.error(f"Audio summarization failed: {result['summary']}")
    
    print(f"\n💡 TRANSCRIPTION OPTIONS:")
    print("   🚀 Azure Whisper API: Fast, cloud-based (requires WHISPER_API_KEY)")
    print("   🔧 Local Whisper: Free, offline (requires model download)")
    print("   🔄 Automatic fallback between methods")
    print(f"\n📊 SUMMARIZATION:")
    print("   📝 Azure OpenAI: High-quality summaries (requires AZURE_API_KEY)")
    print("   ⚡ Automatic configuration based on available credentials")
    print(f"\n🔑 API CONFIGURATION:")
    print(f"   Azure Whisper API: {'✅ Available' if WHISPER_API_KEY else '❌ Not configured'}")
    print(f"   Azure OpenAI API: {'✅ Available' if AZURE_API_KEY else '❌ Not configured'}")
    print(f"   .env file loaded: {'✅ Yes' if DOTENV_AVAILABLE else '❌ Install python-dotenv'}")
