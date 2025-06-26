import os
from types import MappingProxyType
import g4f
from dotenv import load_dotenv


load_dotenv()
env = os.getenv


system_prompt = """🤖✨ You are an expert multilingual AI assistant and developer with extensive experience. Follow these advanced guidelines:

 🌐 1. Language Processing (Intelligent Multilingual Handling) 🧠
     - 🔍 Perform 3-step language analysis:
       1️⃣ 1. Detect primary language using linguistic patterns 🕵️♂️
       2️⃣ 2. Identify secondary languages if code-mixing exceeds 30% 🌍
       3️⃣ 3. Recognize technical terms that should remain untranslated ⚙️
     - 📢 Response language mirroring:
       * 🎯 Match the user's primary language with 98% accuracy
       * 🔒 Preserve original terminology for: proper nouns, technical terms, cultural concepts
       * 🌈 For mixed input (e.g., Hinglish, Spanglish), maintain the dominant language base

 📝 2. Advanced Response Formatting (Structured & Precise) 🎨
     - 🗂 Apply hierarchical organization:
       • 🚀 **<concise 15-word summary>**
       • 📌 Supporting arguments (bullet points)
       • 💻 Examples (indented code blocks if technical)
       • 🌍 Cultural/localization notes (italic when relevant)
     - ⏱ Strict length management:
       * 📏 Real-time character count including Markdown (max 4096)
       * ✂️ Auto-truncation algorithm:
         - 🔄 Preserve complete sentences
         - 🎯 Prioritize core information
         - ➕ Add "[...]" if truncated
     - 🎭 Important style work (other Markdown and emojis):
       * 😊 Use 3-5 relevant emojis per response section
       * 🔀 Use different fonts (MARKDOWN + EMOJI combinations)

 💼 3. Specialized Content Handling ⚙️
     - 👨💻 Technical material:
       > 🔧 Maintain original English terms with localized explanations
       > 💻 Use ```code blocks``` for all commands/APIs
     - 🌏 Cultural adaptation:
       * 📏 Adjust measurements (metric/imperial)
       * 💰 Localize examples (currency, idioms)
       * 🚨 Recognize region-specific sensitivities

 ✅ 4. Quality Assurance Protocols 🔍
     - 🔄 Run pre-response checks:
       1. 📚 Language consistency validation
       2. 📊 Information density audit
       3. 🌐 Cultural appropriateness scan
     - 🧐 Post-generation review:
       * ✔️ Verify factual accuracy
       * 🎚 Ensure tone alignment (professional → friendly spectrum)
       * 📖 Confirm readability score >80%

 📤 Output template:
   ✨ **Title/Subject (if applicable)**
   
   • 🎯 Key point 1
   
   • 🔑 Key point 2
   
   - 📍 Supporting detail
   
   - 💡 Example/excerpt
   
   🌟 Additional tip (optional)
   
 🌍 <cultural/localization note if relevant>



"""



allowed_models = MappingProxyType({
    # DEEPSEEK family
    'Deepseek-R1': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-r1', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_r1, 'client': 'gpt_client'}
        ],
        'api-key': env("DEEPSEEK_API_R1"),
        'code': 'deepseek-r1',
    },
    'Deepseek-V3': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-chat-v3-0324', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_v3, 'client': 'gpt_client'}
        ],
        'code': 'deepseek-v3',
        'api-key': env("DEEPSEEK_API_V3"),
    },
    'Deepseek-R1 (QWEN)': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-r1-distill-qwen-32b', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_r1_distill_qwen_32b, 'client': 'gpt_client'}
        ],
        'code': 'deepseek-r1-qwen',
        'api-key': env("DEEPSEEK_API_QWEN"),
    },


   # GPT family
   'GPT-4 Turbo': {
        'img_support': True,
        'models': [
            {'model': 'gpt-4-turbo', 'client': 'gpt_client'}
        ],
        'code': 'gpt4-turbo',
        'api-key': env("GPT_4_TURBO_API"),
   },
   'GPT-4.1': {
        'img_support': True,
        'models': [
            {'model': 'gpt-4.1', 'client': 'gpt_client'}
        ],
        'code': 'gpt4.1',
        'api-key': env("GPT_4_1_API"),
   },
   'GPT-4o': {
       'img_support': True,
       'models': [
            {'model': 'gpt-4o', 'client': 'gpt_client'}
       ],
       'code': 'gpt4-o',
       'api-key': env("GPT_4_O_API"),
   },

   # MINI GPT`s family
   'GPT-4.1 Mini': {
       'img_support': False,
       'models': [
            {'model': 'gpt-4.1-mini', 'client': 'gpt_client'}
       ],
       'code': 'gpt4.1-mini',
       'api-key': env("GPT_4_1_MINI_API"), 
   },
   'GPT-4o Mini': {
       'img_support': True,
       'models': [
            {'model': 'gpt-4o-mini', 'client': 'gpt_client'}
       ],
       'code': 'gpt4-o-mini',
       'api-key': env("GPT_4_O_MINI_API"),
   },

   # CLAUDE family
   'Claude 3.7 Sonnet': {
       'img_support': True,
       'models': [
            {'model': 'claude-3.7-sonnet', 'client': 'gpt_client'}
       ],
       'code': 'claude3.7-sonnet',
       'api-key': env("CLAUDE_37_API"),
   },
   'Claude 3.7 Sonnet (thinking)': {
       'img_support': True,
       'models': [
            {'model': 'clude-3.7-sonnet-thinking', 'client': 'gpt_client'}
       ],
       'code': 'claude3.7-sonnet-thinking',
       'api-key': env("CLAUDE_37_TH_API"),
   },

   # Open AI family
   'OpenAI o3': {
       'img_support': True,
       'models': [
            {'model': 'openai/o3', 'client': 'openrouter'}
       ],
       'code': 'open-ai-o3',
       'api-key': env("OAI_O3"),
   },
   'Open AI o4 Mini': {
       'img_support': True,
       'models': [
            {'model': g4f.models.o4_mini, 'client': 'gpt_client'}
       ],
       'code': 'open-ai-o4-mini',
       'api-key': env("OAI_O4_MINI"),
   },


   # QWEN family
   'Qwen3 235B A22B': {
       'img_support': False,
       'models': [
            {'model': 'qwen/qwen3-235b-a22b', 'client': 'openrouter'},
            {'model': 'qwen-3-235b', 'client': 'gpt_client'}
       ],
       'code': 'qwen3-235B-A22B',
       'api-key': env("QWEN_3_235"),
   },
   'Qwen3 30B A3B': {
       'img_support': False,
       'models': [
            {'model': 'qwen/qwen3-30b-a3b', 'client': 'openrouter'},
            {'model': 'qwen-3-30b', 'client': 'gpt_client'}
       ],
       'code': 'qwen3-30b-a3b',
       'api-key': env("QWEN_3_30"),
   },



   # Gemini family
   'Gemini 2.0 Flash Lite': {
       'img_support': True,
       'models': [
            {'model': g4f.models.gemini_2_0_flash_thinking, 'client': 'gpt_client'}
       ],
       'code': 'gemini-2.0-flash-lite',
       'api-key': env("GEMINI_API"),
   },
})




img_generation_models = {
    'FLUX': {
        'models': [
            {'model': 'flux', 'client': 'gpt_client'}
        ],
        'versions': [
            {'model': 'FLUX 1.1 [ultra]', 'code': 'flux-11-ultra'},
            {'model': 'FLUX 1.1 dev', 'code': 'flux-11-dev'},

        ],
        'category-code': 'flux',
        'api-key': 'asdasdasd',
    },
}
