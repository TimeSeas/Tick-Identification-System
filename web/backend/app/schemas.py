"""
Pydantic v2 请求/响应模型（与前端 TS 类型一一对应）
通用的 API 响应结构，用于标准化后端接口的返回格式
"""


from typing import Optional
from pydantic import BaseModel

class ApiResponse(BaseModel):
    code:int=200
    msg:str="ok"
    data:Optional[dict]=None
def ok(data=None)->dict :return ApiResponse(data=data).model_dump()
#构造一个成功的响应对象，将传入的 data赋值给 data字段，其余字段保持默认（code=200, msg="ok"）。
# 调用 .model_dump()将模型转为普通字典（JSON 可序列化），方便 FastAPI 等框架直接返回。
#result = {"user_id": 123}
# return ok(result)
# => {"code": 200, "msg": "ok", "data": {"user_id": 123}}


def fail(msg:str,code:int=500)->dict:return ApiResponse(code=code,msg=msg).model_dump()
#构造一个失败的响应对象，必须提供错误消息 msg，可选指定 code（默认 500）。
# 同样转成字典返回。
# return fail("用户不存在", code=404)
# # => {"code": 404, "msg": "用户不存在", "data": None}