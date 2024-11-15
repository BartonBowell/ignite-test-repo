import interactions
import asyncio
import whisper
import numpy as np
import json
import os
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('voice_transcriber')

class VoiceTranscriber(interactions.Client):
    def __init__(self, token):
        super().__init__(token=token)
        
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize Whisper model with CUDA support
        self.whisper_model = whisper.load_model("large").to(device)
        self.device = device
        
        # Ensure recordings directory exists
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        # Add tracking for last processing time
        self.last_processed = datetime.now()
        
        self.is_processing = False
        self.current_voice_state = None  # Add this to track voice state
        
        # Add speaking tracking
        self.is_speaking = False
        self.last_speech_time = time.time()
        
        # Add user cache
        self.user_cache = {}
        
    async def get_username(self, user_id, guild_id):
        """Get username for a given user ID, with caching"""
        cache_key = f"{user_id}_{guild_id}"
        if cache_key in self.user_cache:
            return self.user_cache[cache_key]
            
        try:
            # Try to fetch the member from the guild
            guild = await self.fetch_guild(guild_id)
            member = await guild.fetch_member(user_id)
            username = member.display_name or member.username
            self.user_cache[cache_key] = username
            return username
        except Exception as e:
            logger.error(f"Error fetching username for {user_id}: {str(e)}")
            return f"User_{user_id}"  # Fallback if we can't get the username

    async def cycle_recording(self, voice_state):
        """Cycles the recording based on speech detection"""
        try:
            while self.is_processing:
                try:
                    # Start a new recording
                    new_recording = voice_state.start_recording(
                        output_dir=str(self.recordings_dir),
                        encoding="wav"
                    )
                    await asyncio.sleep(0.1)
                    
                    # Base recording duration
                    recording_duration = 20  # seconds
                    start_time = time.time()
                    
                    # Keep recording while speaking or shortly after speech ends
                    while True:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        time_since_last_speech = current_time - self.last_speech_time
                        
                        # Log the current state for debugging
                        logger.debug(f"Recording state - Speaking: {self.is_speaking}, "
                                   f"Elapsed: {elapsed_time:.1f}s, "
                                   f"Since last speech: {time_since_last_speech:.1f}s")
                        
                        # If speaking, extend duration and continue
                        if self.is_speaking:
                            recording_duration = max(recording_duration, elapsed_time + 2)
                            logger.debug(f"Extended recording duration to {recording_duration}")
                            await asyncio.sleep(0.1)
                            continue
                        
                        # If not speaking, check if we should stop
                        if time_since_last_speech >= 1.0 and elapsed_time >= recording_duration:
                            break
                        
                        await asyncio.sleep(0.1)
                    
                    # Stop the recording and start a new cycle
                    logger.info(f"Stopping recording after {recording_duration:.1f} seconds")
                    await voice_state.stop_recording()
                    await new_recording
                    
                except Exception as e:
                    logger.error(f"Error in recording cycle: {str(e)}")
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error in cycle_recording: {str(e)}")
    
    @interactions.listen()
    async def on_voice_state_update(self, event):
        """Track when users start/stop speaking"""
        if event.after and event.after.user.id == self.user.id:
            return  # Ignore bot's own voice state
            
        if event.after and event.after.speaking:
            self.is_speaking = True
            self.last_speech_time = time.time()
            logger.debug("User started speaking")
        else:
            self.is_speaking = False
            logger.debug("User stopped speaking")
    
    @interactions.slash_command(name="join", description="Join your voice channel and start transcribing")
    async def join(self, ctx: interactions.SlashContext):
        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel first!")
            return
        
        # Stop any existing processing
        self.is_processing = False
        await asyncio.sleep(1)
        
        # Start new processing session
        self.is_processing = True
        
        try:
            voice_state = await ctx.author.voice.channel.connect()
            self.current_voice_state = voice_state
            await ctx.send(f"Joined {ctx.author.voice.channel.name}")
            
            # Start recording
            await voice_state.start_recording(
                output_dir=str(self.recordings_dir),
                encoding="wav"
            )
            
            # Start both tasks
            asyncio.create_task(self.continuous_processing(ctx))
            asyncio.create_task(self.cycle_recording(voice_state))
            
        except Exception as e:
            logger.error(f"Error in join command: {str(e)}")
            await ctx.send("Failed to join voice channel. Please try again.")
    
    async def continuous_processing(self, ctx):
        """Continuously process recordings"""
        logger.info("Starting continuous processing")
        while self.is_processing:
            try:
                files = list(self.recordings_dir.glob("*.wav"))
                for file_path in files:
                    try:
                        file_age = time.time() - file_path.stat().st_mtime
                        # Only process files that are complete (not being written to)
                        if file_age > 0.75 and not self.is_speaking:
                            file_size = file_path.stat().st_size
                            if file_size > 1024:
                                logger.info(f"Processing file {file_path} (age: {file_age}s, size: {file_size})")
                                result = self.whisper_model.transcribe(
                                    str(file_path),
                                    language="en",
                                    task="transcribe",
                                    condition_on_previous_text=False,
                                    prompt="Speak naturally.",
                                    temperature=0.0,
                                    compression_ratio_threshold=1.5,
                                    no_speech_threshold=0.6,
                                    fp16=True  # Enable half-precision for faster GPU processing
                                )
                                
                                transcription = result["text"].strip()
                                if transcription:
                                    # Extract user ID from filename
                                    user_id = file_path.stem.split('_')[1] if '_' in file_path.stem else file_path.stem
                                    # Get username
                                    username = await self.get_username(user_id, ctx.guild_id)
                                    await ctx.channel.send(f"{username}: {transcription}")
                            
                            if file_path.exists():
                                file_path.unlink()
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        if file_path.exists():
                            file_path.unlink()
                
            except Exception as e:
                logger.error(f"Error in continuous processing: {str(e)}")
            
            await asyncio.sleep(0.5)
    
    @interactions.slash_command(name="leave", description="Leave the voice channel")
    async def leave(self, ctx: interactions.SlashContext):
        if not ctx.voice_state:
            await ctx.send("I'm not in a voice channel!")
            return
        
        self.is_processing = False
        await asyncio.sleep(1)
        
        if self.current_voice_state:
            await self.current_voice_state.stop_recording()
            await self.current_voice_state.disconnect()
            self.current_voice_state = None
        
        await ctx.send("Left the voice channel")

def main():
    # Load configuration
    if not os.path.exists('config.json'):
        with open('config.json', 'w') as f:
            json.dump({
                'token': 'YOUR_BOT_TOKEN_HERE'
            }, f, indent=4)
        print("Created config.json template. Please add your Discord bot token.")
        return
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    bot = VoiceTranscriber(token=config['token'])
    bot.start()

if __name__ == "__main__":
    main()