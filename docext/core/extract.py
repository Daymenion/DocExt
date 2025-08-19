from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from typing import Union

import json_repair
import mdpd
import pandas as pd
from loguru import logger

from docext.core.client import sync_request
from docext.core.confidence import get_fields_confidence_score_messages_binary
from docext.core.prompts import get_fields_messages
from docext.core.prompts import get_tables_messages
from docext.core.utils import convert_files_to_images
from docext.core.utils import resize_images
from docext.core.utils import validate_fields_and_tables
from docext.core.utils import validate_file_paths


def extract_fields_from_documents(
    file_paths: list[str],
    model_name: str,
    fields: list[dict],
):
    """
    Extract specified fields from documents using a vision-language model.
    
    Args:
        file_paths: List of paths to document files
        model_name: Name of the VLM model to use
        fields: List of field dictionaries with 'name' and optional 'description'
        
    Returns:
        pandas.DataFrame: Extracted fields with confidence scores
        
    Raises:
        ValueError: If fields list is invalid
        Exception: If extraction fails
    """
    if len(fields) == 0:
        logger.warning("No fields specified for extraction")
        return pd.DataFrame()
        
    try:
        field_names = [field["name"] for field in fields]
        fields_description = [field.get("description", "") for field in fields]
        
        logger.debug(f"Extracting fields: {field_names}")
        messages = get_fields_messages(field_names, fields_description, file_paths)

        format_fields = {
            "type": "object",
            "properties": {field_name: {"type": "string"} for field_name in field_names},
        }

        logger.info(f"Sending field extraction request to {model_name}")
        response = sync_request(messages, model_name, format=format_fields)["choices"][0][
            "message"
        ]["content"]
        logger.debug(f"Field extraction response: {response}")

        # Get confidence scores
        messages = get_fields_confidence_score_messages_binary(
            messages,
            response,
            field_names,
        )

        format_fields_conf_score = {
            "type": "object",
            "properties": {
                field_name: {"type": "string", "enum": ["High", "Low"]}
                for field_name in field_names
            },
        }

        logger.info(f"Requesting confidence scores from {model_name}")
        response_conf_score = sync_request(
            messages,
            model_name,
            format=format_fields_conf_score,
        )["choices"][0]["message"]["content"]
        logger.debug(f"Confidence score response: {response_conf_score}")

        # Parse responses with error handling
        try:
            extracted_fields = json_repair.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse extracted fields: {e}")
            raise ValueError(f"Invalid field extraction response format: {response}")
            
        try:
            conf_scores = json_repair.loads(response_conf_score)
        except Exception as e:
            logger.error(f"Failed to parse confidence scores: {e}")
            # Create default confidence scores if parsing fails
            conf_scores = {field: "Low" for field in field_names}

        logger.info(f"Extracted fields: {extracted_fields}")
        logger.info(f"Confidence scores: {conf_scores}")

        # Handle both single dictionary and list of dictionaries
        if not isinstance(extracted_fields, list):
            extracted_fields = [extracted_fields]
        
        # Handle confidence scores similarly
        if not isinstance(conf_scores, list):
            conf_scores = [conf_scores] * len(extracted_fields)
        elif len(conf_scores) < len(extracted_fields):
            # If we have fewer confidence scores than documents, pad with the first confidence score
            conf_scores.extend([conf_scores[0]] * (len(extracted_fields) - len(conf_scores)))
        
        # Create a list of dataframes, one for each document
        dfs = []
        for idx, (doc_fields, doc_conf_scores) in enumerate(zip(extracted_fields, conf_scores)):
            df = pd.DataFrame(
                {
                    "fields": field_names,
                    "answer": [doc_fields.get(field, "") for field in field_names],
                    "confidence": [doc_conf_scores.get(field, "Low") for field in field_names],
                    "document_index": [idx] * len(field_names)
                },
            )
            dfs.append(df)
        
        # Concatenate all dataframes with a document index
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
        else:
            # Return empty DataFrame with correct columns if no data
            final_df = pd.DataFrame(columns=["fields", "answer", "confidence", "document_index"])
        
        return final_df
        
    except Exception as e:
        logger.error(f"Field extraction failed: {e}")
        # Return empty DataFrame with correct columns on error
        return pd.DataFrame(columns=["fields", "answer", "confidence", "document_index"])


def extract_tables_from_documents(
    file_paths: list[str],
    model_name: str,
    columns: list[dict],
):
    """
    Extract tables from documents using a vision-language model.
    
    Args:
        file_paths: List of paths to document files
        model_name: Name of the VLM model to use
        columns: List of column dictionaries with 'name', 'type', and optional 'description'
        
    Returns:
        pandas.DataFrame: Extracted table data
        
    Raises:
        ValueError: If columns list is invalid or response format is incorrect
        Exception: If extraction fails
    """
    if len(columns) == 0:
        logger.warning("No columns specified for table extraction")
        return pd.DataFrame()
        
    try:
        columns_names = [column["name"] for column in columns if column["type"] == "table"]
        if not columns_names:
            logger.warning("No table columns found in columns list")
            return pd.DataFrame()
            
        columns_description = [
            column.get("description", "") for column in columns if column["type"] == "table"
        ]
        
        logger.debug(f"Extracting table columns: {columns_names}")
        messages = get_tables_messages(columns_names, columns_description, file_paths)

        logger.info(f"Sending table extraction request to {model_name}")
        response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
        logger.debug(f"Table extraction response: {response}")

        # Extract markdown table from response
        try:
            if "|" not in response:
                logger.error("No table markers found in response")
                raise ValueError("Response does not contain a valid markdown table")
                
            table_start = response.index("|")
            table_end = response.rindex("|") + 1
            table_md = response[table_start:table_end]
            
            logger.debug(f"Extracted table markdown: {table_md}")
            
            # Convert markdown table to DataFrame
            df = mdpd.from_md(table_md)
            
            if df.empty:
                logger.warning("Extracted table is empty")
                
            return df
            
        except ValueError as e:
            logger.error(f"Failed to extract table from response: {e}")
            raise ValueError(f"Invalid table format in response: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to parse markdown table: {e}")
            raise ValueError(f"Could not parse table from response: {str(e)}")
            
    except Exception as e:
        logger.error(f"Table extraction failed: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame()


def extract_information(
    file_inputs: list[tuple],
    model_name: str,
    max_img_size: int,
    fields_and_tables: dict[str, list[dict]] | pd.DataFrame,
):
    """
    Extract information from documents with optimized processing.
    
    Args:
        file_inputs: List of file inputs (tuples or paths)
        model_name: Name of the VLM model to use
        max_img_size: Maximum image size for processing
        fields_and_tables: Fields and tables configuration
        
    Returns:
        tuple: (fields_df, tables_df) - DataFrames with extracted data
        
    Raises:
        ValueError: If inputs are invalid
        Exception: If extraction fails
    """
    try:
        # Validate and process inputs
        fields_and_tables = validate_fields_and_tables(fields_and_tables)
        
        if len(fields_and_tables["fields"]) == 0 and len(fields_and_tables["tables"]) == 0:
            logger.warning("No fields or tables specified for extraction")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract file paths
        file_paths: list[str] = [
            file_input[0] if isinstance(file_input, tuple) else file_input
            for file_input in file_inputs
        ]
        
        if not file_paths:
            logger.warning("No files provided for extraction")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Processing {len(file_paths)} files with model '{model_name}'")
        
        # Validate file paths
        validate_file_paths(file_paths)
        
        # Convert and resize images
        file_paths = convert_files_to_images(file_paths)
        resize_images(file_paths, max_img_size)
        
        # Determine what to extract
        extract_fields = len(fields_and_tables["fields"]) > 0
        extract_tables = len(fields_and_tables["tables"]) > 0
        
        fields_df = pd.DataFrame()
        tables_df = pd.DataFrame()
        
        # Use parallel processing if both fields and tables need extraction
        if extract_fields and extract_tables:
            logger.debug("Extracting fields and tables in parallel")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_fields = executor.submit(
                    extract_fields_from_documents,
                    file_paths,
                    model_name,
                    fields_and_tables["fields"],
                )
                future_tables = executor.submit(
                    extract_tables_from_documents,
                    file_paths,
                    model_name,
                    fields_and_tables["tables"],
                )
                
                fields_df = future_fields.result()
                tables_df = future_tables.result()
        
        # Extract only fields if tables not needed
        elif extract_fields:
            logger.debug("Extracting fields only")
            fields_df = extract_fields_from_documents(
                file_paths,
                model_name,
                fields_and_tables["fields"],
            )
        
        # Extract only tables if fields not needed
        elif extract_tables:
            logger.debug("Extracting tables only")
            tables_df = extract_tables_from_documents(
                file_paths,
                model_name,
                fields_and_tables["tables"],
            )
        
        # Post-process results
        if not fields_df.empty and 'document_index' in fields_df.columns:
            fields_df = fields_df.sort_values(['document_index', 'fields'])
            logger.debug(f"Extracted {len(fields_df)} field records")
        
        if not tables_df.empty:
            logger.debug(f"Extracted table with {len(tables_df)} rows")
        
        return fields_df, tables_df
        
    except Exception as e:
        logger.error(f"Information extraction failed: {e}")
        return pd.DataFrame(), pd.DataFrame()
