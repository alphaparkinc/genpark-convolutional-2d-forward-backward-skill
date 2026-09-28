import json
import sys
from client import Conv2D

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "conv2d_forward",
                        "description": "Compute 2D spatial convolution on input matrix with kernel",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "image": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "kernel": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "stride": {"type": "integer", "default": 1},
                                "padding": {"type": "integer", "default": 0}
                            },
                            "required": ["image", "kernel"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "conv2d_forward":
            res = Conv2D.forward(args["image"], args["kernel"], args.get("stride", 1), args.get("padding", 0))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"output": res})}]}
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
