# Document Upload Feature

This chatbot now supports uploading PDF and document files that are stored in AWS S3.

## Supported File Types
- PDF (`.pdf`)
- Microsoft Word (`.doc`, `.docx`)
- Plain Text (`.txt`)

## Maximum File Size
- 10 MB per file

## Setup Instructions

### 1. AWS S3 Setup
1. Create an AWS S3 bucket
2. Configure appropriate bucket permissions
3. Create an IAM user with S3 access
4. Generate Access Key ID and Secret Access Key

### 2. Environment Variables
Create a `.env.local` file in the root directory with the following variables:

```env
# RAG API Configuration (optional)
RAG_API_URL=

# LangSmith Configuration (for tracing and monitoring)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=chatbot-project

# AWS S3 Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_S3_BUCKET_NAME=your_bucket_name

# OpenAI Configuration (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# Anthropic Configuration (if using Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key

# General App Configuration
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### 3. S3 Bucket Policy (Example)
Your S3 bucket should have a policy that allows your IAM user to put and get objects:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR_ACCOUNT_ID:user/YOUR_IAM_USER"
      },
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```

### 4. IAM User Permissions
Attach a policy to your IAM user with the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```

## Usage

1. Click the 📎 (paperclip) button in the chat interface
2. Select a PDF or document file
3. The file will be uploaded to S3 and displayed in the uploaded files list
4. You can remove files before sending your message
5. When you send a message, the uploaded files will be included in the context

## API Endpoints

### Upload File
- **Endpoint**: `POST /api/upload`
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with `file` field
- **Response**:
```json
{
  "success": true,
  "file": {
    "name": "document.pdf",
    "size": 12345,
    "type": "application/pdf",
    "key": "uploads/1234567890-document.pdf",
    "url": "https://your-bucket.s3.amazonaws.com/..."
  }
}
```

## File Storage

Files are stored in S3 with the following structure:
- Path: `uploads/{timestamp}-{filename}`
- Pre-signed URLs are generated with 1-hour expiration
- Files are accessible through secure pre-signed URLs

## Security Considerations

1. **Environment Variables**: Never commit `.env.local` to version control
2. **File Validation**: Only allowed file types are accepted
3. **File Size Limits**: Maximum 10MB per file
4. **Pre-signed URLs**: URLs expire after 1 hour for security
5. **IAM Permissions**: Use least-privilege principle for IAM user

## Future Enhancements

- [ ] Extract text from PDFs for RAG processing
- [ ] Support for more file types
- [ ] File content search and indexing
- [ ] Multiple file upload at once
- [ ] File preview functionality
- [ ] Upload progress indicator
