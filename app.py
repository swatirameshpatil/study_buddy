
import chainlit as cl
import os
import tempfile
from src.StudyBuddy import StudyBuddy
from src.logger import get_logger

logger = get_logger(__name__)

buddy = StudyBuddy()

@cl.on_chat_start
async def start():
    logger.info(f"Chat started by user: {cl.user_session.get("id")}")
    await cl.Message(content="Study Buddy ready! Upload your documents").send()

@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Message received from user {cl.user_session.get("id")}: {message.content[:50] if message.content else 'File upload'}")
    
    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                try:
                    logger.info(f"Processing file upload: {element.name}")
                    stats = buddy.process_upload(element.path, cl.user_session.get("id"))
                    logger.info(f"File processed successfully: {stats['filename']} - {stats['chunks_count']} chunks")
                    await cl.Message(
                        content=f"✅ Indexed **{stats['chunks_count']} chunks** from **{stats['filename']}**!\n"
                               f"📊 {stats['raw_pages']} pages → ready for Q&A."
                    ).send()
                except Exception as e:
                    logger.error(f"Error processing file {element.name}: {str(e)}", exc_info=True)
                    await cl.Message(content=f"Error processing file: {str(e)}").send()
    
    # Handle text messages for Q&A
    elif message.content:
        logger.info(f"Retrieving documents for query: {message.content[:50]}")
        docs = buddy.retrieve(message.content, cl.user_session.get("id"))
        logger.info(f"Found {len(docs)} relevant chunks")
        await cl.Message(content=f"Found {len(docs)} relevant chunks. Q&A tomorrow!").send()