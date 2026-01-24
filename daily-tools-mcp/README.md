# Daily Tools MCP Server

An MCP (Model Context Protocol) server that provides useful daily tools for developers. Integrates with Claude Code to give AI access to JSON/YAML processing, image manipulation, PDF utilities, code formatting, HTTP requests, and text processing.

## Features

### JSON/YAML Processing
| Tool | Description |
|------|-------------|
| `jq_query` | Query/transform JSON using jq syntax |
| `yq_query` | Query/transform YAML using yq syntax |
| `json_to_yaml` | Convert JSON to YAML |
| `yaml_to_json` | Convert YAML to JSON |
| `json_format` | Pretty-print or minify JSON |

### Image Processing (ImageMagick)
| Tool | Description |
|------|-------------|
| `image_resize` | Resize images to specific dimensions |
| `image_convert` | Convert between formats (png, jpg, webp) |
| `image_compress` | Optimize/compress images |
| `image_info` | Get image metadata |

### PDF Utilities (poppler)
| Tool | Description |
|------|-------------|
| `pdf_to_text` | Extract text from PDF |
| `pdf_info` | Get PDF metadata |
| `pdf_merge` | Merge multiple PDFs |
| `pdf_split` | Extract specific pages |

### Code Formatting
| Tool | Description |
|------|-------------|
| `format_python` | Format Python code with Black |
| `format_js` | Format JS/TS with Prettier |
| `format_json` | Format JSON with indentation |
| `format_sql` | Format SQL with sqlfluff |
| `lint_python` | Check Python code style |

### HTTP Utilities
| Tool | Description |
|------|-------------|
| `http_get` | GET request with headers |
| `http_post` | POST request with JSON body |
| `download_file` | Download file to path |
| `check_url` | Check if URL is reachable |

### Text Processing
| Tool | Description |
|------|-------------|
| `base64_encode` | Encode text/file to base64 |
| `base64_decode` | Decode base64 to text/file |
| `hash_text` | Generate MD5/SHA256 hash |
| `hash_file` | Hash file contents |
| `word_count` | Count words, lines, chars |
| `regex_extract` | Extract matches with regex |
| `regex_replace` | Replace with regex |
| `diff_text` | Show diff between texts |

## Installation

### 1. Install System Dependencies

```bash
# macOS
brew install jq yq imagemagick poppler
npm install -g prettier

# Ubuntu/Debian
sudo apt install jq yq imagemagick poppler-utils
npm install -g prettier
```

### 2. Install the MCP Server

```bash
cd daily-tools-mcp
uv sync
```

### 3. Register with Claude Code

```bash
claude mcp add daily-tools -- uv run --directory /path/to/daily-tools-mcp daily-tools
```

Or add manually to your `.mcp.json`:

```json
{
  "mcpServers": {
    "daily-tools": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/daily-tools-mcp", "daily-tools"]
    }
  }
}
```

## Usage Examples

Once registered, you can ask Claude Code to use these tools:

**JSON Processing:**
```
"Use jq to extract all names from: {\"users\":[{\"name\":\"Alice\"},{\"name\":\"Bob\"}]}"
"Convert this YAML to JSON: ..."
```

**Image Processing:**
```
"Resize image.png to 800px wide and save as image_small.png"
"Convert photo.jpg to webp format with 75% quality"
"What are the dimensions of this image?"
```

**PDF Processing:**
```
"Extract text from document.pdf"
"Merge report1.pdf and report2.pdf into combined.pdf"
"How many pages are in this PDF?"
```

**Code Formatting:**
```
"Format this Python code: def foo(x):return x+1"
"Format this SQL query: select * from users where id=1"
```

**HTTP Requests:**
```
"Check if https://api.github.com is reachable"
"Make a GET request to https://httpbin.org/json"
```

**Text Processing:**
```
"Encode this text to base64: Hello World"
"Generate SHA256 hash of this text: secret"
"Count the words in this text: ..."
"Extract all email addresses from this text using regex"
```

## Development

### Run the server manually

```bash
uv run daily-tools
```

### Project Structure

```
daily-tools-mcp/
├── pyproject.toml
├── src/
│   └── daily_tools/
│       ├── __init__.py
│       ├── server.py          # Main MCP server
│       ├── utils.py           # Shared utilities
│       └── tools/
│           ├── json_yaml.py   # jq/yq tools
│           ├── images.py      # ImageMagick tools
│           ├── pdf.py         # poppler tools
│           ├── code.py        # formatting tools
│           ├── http.py        # HTTP tools
│           └── text.py        # text processing
└── README.md
```

## Requirements

- Python 3.10+
- uv package manager
- System tools: jq, yq, imagemagick, poppler-utils
- Node.js (for prettier)

## License

MIT
