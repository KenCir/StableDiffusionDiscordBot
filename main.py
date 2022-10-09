import threading
import asyncio
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import discord
from discord.ext import commands

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = "stable-diffusion token"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16,
                                                   use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

using_status = False


def image_generete(ctx, args):
    global using_status

    with autocast(DEVICE):
        image = pipe(args, guidance_scale=7.5)["sample"][0]
        image.save("test.png")
        bot.loop.create_task(ctx.reply(file=discord.File("test.png")))
        using_status = False

@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')
    await bot.change_presence(activity=discord.Game(name="Stable Diffusion"))


@bot.command()
async def image(ctx, *args):
    global using_status

    if using_status:
        await ctx.reply(f'他の画像を生成中です')
        return

    using_status = True
    await ctx.reply(f'{" ".join(args)}で画像を生成します、お待ちください...')
    thread = threading.Thread(target=image_generete, args=(ctx, " ".join(args)))
    thread.start()


bot.run("discord TOKEN")
