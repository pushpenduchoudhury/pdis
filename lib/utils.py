import base64

def key_encode(string):
    base64_bytes = str(string).encode("ascii")
    encode_string_bytes = base64.b64encode(base64_bytes)
    encoded_string = encode_string_bytes.decode("ascii")
    return encoded_string

def key_decode(encoded_string):
    base64_bytes = str(encoded_string).encode("ascii")
    decode_string_bytes = base64.b64decode(base64_bytes)
    decoded_string = decode_string_bytes.decode("ascii")
    return decoded_string

