import streamlit as st
import json
from groq import Groq
import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
import toml
from toolhouse import Toolhouse
import google.generativeai as genai
from streamlit_js_eval import streamlit_js_eval, get_geolocation
import requests as http_requests
from datetime import datetime, timedelta
import urllib.parse

# ============== GAMIFICATION SYSTEM ==============

BADGES = {
    "first_scan": {"name": "🌱 First Step", "desc": "Scanned your first item", "points_req": 0, "icon": "🌱"},
    "recycler_10": {"name": "♻️ Eco Starter", "desc": "Recycled 10 items", "points_req": 100, "icon": "♻️"},
    "recycler_50": {"name": "🌿 Green Warrior", "desc": "Recycled 50 items", "points_req": 500, "icon": "🌿"},
    "recycler_100": {"name": "🌳 Earth Champion", "desc": "Recycled 100 items", "points_req": 1000, "icon": "🌳"},
    "streak_3": {"name": "🔥 On Fire", "desc": "3 day streak", "points_req": 0, "icon": "🔥"},
    "streak_7": {"name": "⚡ Week Warrior", "desc": "7 day streak", "points_req": 0, "icon": "⚡"},
    "streak_30": {"name": "💎 Monthly Master", "desc": "30 day streak", "points_req": 0, "icon": "💎"},
    "plastic_hero": {"name": "🥤 Plastic Hero", "desc": "Recycled 20 plastic items", "points_req": 0, "icon": "🥤"},
    "paper_saver": {"name": "📄 Paper Saver", "desc": "Recycled 20 paper items", "points_req": 0, "icon": "📄"},
    "e_waste_pro": {"name": "💻 E-Waste Pro", "desc": "Properly disposed 5 electronics", "points_req": 0, "icon": "💻"},
    "location_finder": {"name": "🗺️ Explorer", "desc": "Found recycling facilities", "points_req": 0, "icon": "🗺️"},
}

def init_gamification():
    """Initialize gamification state"""
    if 'game_stats' not in st.session_state:
        st.session_state.game_stats = {
            'total_points': 0,
            'items_recycled': 0,
            'current_streak': 0,
            'longest_streak': 0,
            'last_scan_date': None,
            'badges_earned': [],
            'category_counts': {
                'plastic': 0,
                'paper': 0,
                'metal': 0,
                'glass': 0,
                'organic': 0,
                'e_waste': 0,
                'other': 0
            },
            'weekly_items': [],
            'level': 1,
            'xp_to_next_level': 100
        }

def get_level_info(points):
    """Calculate level based on points"""
    level = 1 + points // 100
    xp_in_level = points % 100
    xp_to_next = 100
    return level, xp_in_level, xp_to_next

def categorize_item(item_name):
    """Categorize item for tracking"""
    item_lower = item_name.lower()
    if any(w in item_lower for w in ['plastic', 'bottle', 'container', 'pet', 'wrapper', 'bag']):
        return 'plastic'
    elif any(w in item_lower for w in ['paper', 'cardboard', 'newspaper', 'magazine', 'book']):
        return 'paper'
    elif any(w in item_lower for w in ['metal', 'can', 'aluminum', 'tin', 'steel']):
        return 'metal'
    elif any(w in item_lower for w in ['glass', 'jar']):
        return 'glass'
    elif any(w in item_lower for w in ['food', 'organic', 'peel', 'fruit', 'vegetable', 'compost']):
        return 'organic'
    elif any(w in item_lower for w in ['electronic', 'battery', 'phone', 'computer', 'cable', 'charger', 'electrical', 'plug', 'wire', 'adapter', 'outlet']):
        return 'e_waste'
    return 'other'

def get_correct_bin_for_item(item_name, full_advice):
    """Determine the correct bin by parsing the actual AI advice for this item"""
    item_lower = item_name.lower()
    
    # Find the section of advice that talks about this specific item
    item_section = ""
    lines = full_advice.split('\n')
    capturing = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check if this line mentions our item (item name header)
        if item_lower in line_lower and ('item name' in line_lower or item_lower == line_lower.strip().rstrip(':')):
            capturing = True
            item_section = ""
            continue
        elif capturing:
            # Stop capturing when we hit the next item
            if 'item name' in line_lower and item_lower not in line_lower:
                break
            item_section += " " + line
    
    # If we didn't find a specific section, search around the item mention
    if not item_section:
        for i, line in enumerate(lines):
            if item_lower in line.lower():
                # Get surrounding context (5 lines after)
                item_section = " ".join(lines[i:i+6])
                break
    
    search_text = item_section.lower() if item_section else full_advice.lower()
    
    # Parse the actual bin color from the advice
    # Look for explicit "Correct Bin:" mentions first
    if 'correct bin:' in search_text:
        # Extract just the bin specification
        bin_part = search_text.split('correct bin:')[1]
        # Get until next line or bullet
        bin_line = bin_part.split('\n')[0].split('-')[0].strip()
        
        # Check for yellow_street FIRST (before yellow)
        if 'yellow_street' in bin_line or 'yellow street' in bin_line:
            return 'special'  # yellow_street is for textiles/donation
        elif 'red' in bin_line:
            return 'red'
        elif 'grey' in bin_line or 'gray' in bin_line:
            return 'grey'
        elif 'yellow' in bin_line:
            return 'yellow'
        elif 'blue' in bin_line:
            return 'blue'
        elif 'brown' in bin_line or 'organic' in bin_line:
            return 'brown'
        elif 'green' in bin_line:
            return 'green'
        elif 'special' in bin_line or 'collection' in bin_line or 'donate' in bin_line:
            return 'special'
    
    # Secondary detection from context - check yellow_street FIRST
    if 'yellow_street' in search_text or 'yellow street' in search_text:
        return 'special'  # Street containers for textiles
    if 'textile' in search_text or 'clothing' in search_text or 'fabric' in search_text:
        return 'special'
    if 'donate' in search_text or 'donation' in search_text or 'charity' in search_text:
        return 'special'
    if 'collection point' in search_text or 'collection center' in search_text:
        return 'special'
    if 'optician' in search_text or 'optical' in search_text or 'eyewear' in search_text:
        return 'special'
    if 'sweatshirt' in item_name.lower() or 'hoodie' in item_name.lower() or 'shirt' in item_name.lower():
        # Clothing items always go to textile/donation
        if 'donate' in search_text or 'textile' in search_text or 'yellow_street' in search_text or 'clothing' in search_text:
            return 'special'
    if 'glasses' in item_name.lower() or 'eyeglasses' in item_name.lower():
        # Glasses for donation typically
        if 'donate' in search_text or 'optician' in search_text or 'special' in search_text:
            return 'special'
    if 'e-waste' in search_text or 'electronic' in search_text:
        return 'red'
    if 'grey' in search_text or 'gray' in search_text or 'unsorted' in search_text or 'general waste' in search_text:
        return 'grey'
    if 'blue' in search_text and 'paper' in search_text:
        return 'blue'
    if 'brown' in search_text or 'organic' in search_text or 'compost' in search_text:
        return 'brown'
    if 'green' in search_text and 'glass' in search_text:
        return 'green'
    if 'yellow' in search_text and ('plastic' in search_text or 'metal' in search_text):
        return 'yellow'
    if 'red' in search_text:
        return 'red'
    
    # Fallback based on item type
    category = categorize_item(item_name)
    defaults = {
        'plastic': 'yellow',
        'paper': 'blue',
        'metal': 'yellow',
        'glass': 'green',
        'organic': 'brown',
        'e_waste': 'red',
        'other': 'grey'
    }
    return defaults.get(category, 'grey')

def get_correct_bin(item_name, recycling_advice):
    """Wrapper for backward compatibility"""
    return get_correct_bin_for_item(item_name, recycling_advice)

def create_pending_verification(items_str, recycling_advice):
    """Create a pending verification challenge"""
    init_gamification()
    
    items = [i.strip() for i in items_str.split(',') if i.strip()]
    
    # Determine correct bins for each item
    item_bins = {}
    for item in items:
        correct_bin = get_correct_bin(item, recycling_advice)
        item_bins[item] = correct_bin
        # Debug: print what was detected
        print(f"DEBUG: Item '{item}' -> Bin '{correct_bin}'")
    
    # Debug: Print full advice for reference
    print(f"DEBUG: Full advice:\n{recycling_advice[:500]}...")
    
    st.session_state.pending_verification = {
        'items': items,
        'item_bins': item_bins,
        'created_at': datetime.now().isoformat(),
        'advice': recycling_advice
    }

def verify_and_award_points(selected_bins):
    """Verify user's bin selections and award points"""
    init_gamification()
    
    if 'pending_verification' not in st.session_state:
        return 0, [], False
    
    pending = st.session_state.pending_verification
    stats = st.session_state.game_stats
    
    # Check answers
    correct_count = 0
    total_items = len(pending['items'])
    
    for item, correct_bin in pending['item_bins'].items():
        user_bin = selected_bins.get(item, '')
        if user_bin.lower() == correct_bin.lower():
            correct_count += 1
    
    # Calculate points based on accuracy
    accuracy = correct_count / total_items if total_items > 0 else 0
    
    if accuracy >= 0.8:  # At least 80% correct to earn points
        # Full points for correct answers
        points_earned = correct_count * 10
        bonus = 5 if accuracy == 1.0 else 0  # Perfect score bonus
        total_points = points_earned + bonus
        
        stats['total_points'] += total_points
        stats['items_recycled'] += correct_count
        
        # Update category counts only for correct items
        for item, correct_bin in pending['item_bins'].items():
            if selected_bins.get(item, '').lower() == correct_bin.lower():
                category = categorize_item(item)
                stats['category_counts'][category] += 1
        
        # Update streak
        today = datetime.now().date()
        if stats['last_scan_date']:
            last_date = datetime.strptime(stats['last_scan_date'], '%Y-%m-%d').date()
            if today != last_date:
                if today - last_date == timedelta(days=1):
                    stats['current_streak'] += 1
                elif today - last_date > timedelta(days=1):
                    stats['current_streak'] = 1
        else:
            stats['current_streak'] = 1
        
        stats['last_scan_date'] = today.strftime('%Y-%m-%d')
        stats['longest_streak'] = max(stats['longest_streak'], stats['current_streak'])
        stats['level'], _, stats['xp_to_next_level'] = get_level_info(stats['total_points'])
        
        # Check for new badges
        new_badges = check_badges()
        
        # Clear pending verification
        del st.session_state.pending_verification
        
        return total_points, new_badges, True
    else:
        # Not enough correct - no points, show correct answers
        return 0, [], False

def add_points(items_str):
    """Legacy function - now just creates pending verification"""
    # This is now handled by create_pending_verification
    pass
    
    return points_earned, new_badges

def get_youtube_recycling_videos(items):
    """Generate YouTube video suggestions for recyclable items"""
    video_suggestions = []
    
    # Keywords to enhance search relevance
    recycling_keywords = {
        'plastic': ['DIY plastic recycling', 'upcycle plastic bottle', 'plastic bottle crafts'],
        'paper': ['paper recycling at home', 'DIY recycled paper', 'paper crafts reuse'],
        'cardboard': ['cardboard upcycling ideas', 'DIY cardboard projects', 'recycle cardboard crafts'],
        'glass': ['glass bottle upcycling', 'DIY glass jar crafts', 'reuse glass bottles'],
        'metal': ['metal can crafts', 'upcycle tin cans', 'aluminum recycling DIY'],
        'can': ['tin can DIY projects', 'upcycle aluminum cans', 'can crafts ideas'],
        'bottle': ['bottle upcycling ideas', 'DIY bottle crafts', 'reuse bottles creatively'],
        'container': ['container upcycling', 'reuse food containers', 'DIY container crafts'],
        'organic': ['composting at home', 'DIY compost bin', 'food waste composting'],
        'food': ['composting food scraps', 'reduce food waste', 'kitchen composting'],
        'electronic': ['e-waste recycling guide', 'recycle old electronics', 'e-waste disposal tips'],
        'battery': ['battery recycling safely', 'dispose batteries properly', 'battery recycling guide'],
        'clothes': ['upcycle old clothes', 'DIY clothing recycling', 'repurpose old clothes'],
        'fabric': ['fabric upcycling ideas', 'reuse old fabric', 'textile recycling DIY'],
    }
    
    for item in items:
        item_lower = item.lower().strip()
        
        # Find matching category
        search_terms = []
        matched = False
        
        for category, keywords in recycling_keywords.items():
            if category in item_lower:
                search_terms = keywords[:2]  # Get top 2 search terms
                matched = True
                break
        
        # If no category match, create generic search
        if not matched:
            search_terms = [
                f"how to recycle {item}",
                f"upcycle {item} DIY"
            ]
        
        # Generate YouTube search URLs
        for term in search_terms:
            encoded_query = urllib.parse.quote(term)
            youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            video_suggestions.append({
                'item': item,
                'search_term': term,
                'url': youtube_url
            })
    
    return video_suggestions

def display_youtube_suggestions(items):
    """Display YouTube video suggestions in a nice format"""
    suggestions = get_youtube_recycling_videos(items)
    
    if not suggestions:
        return
    
    st.markdown("---")
    st.subheader("🎬 Learn How to Recycle These Items")
    st.caption("Watch helpful YouTube videos about recycling and upcycling")
    
    # Group suggestions by item
    items_shown = set()
    
    for suggestion in suggestions:
        item = suggestion['item']
        
        # Only show one video suggestion per item
        if item in items_shown:
            continue
        items_shown.add(item)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**♻️ {item.title()}**")
            st.markdown(f"🔍 *{suggestion['search_term']}*")
        
        with col2:
            st.link_button("▶️ Watch Videos", suggestion['url'], use_container_width=True)
    
    # Add a general recycling tips video link
    st.markdown("---")
    st.markdown("**📚 Want to learn more about recycling?**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        general_url = "https://www.youtube.com/results?search_query=recycling+tips+for+beginners"
        st.link_button("🌱 Beginner Tips", general_url, use_container_width=True)
    
    with col2:
        diy_url = "https://www.youtube.com/results?search_query=creative+upcycling+ideas"
        st.link_button("🎨 Upcycling Ideas", diy_url, use_container_width=True)
    
    with col3:
        zero_waste_url = "https://www.youtube.com/results?search_query=zero+waste+lifestyle+tips"
        st.link_button("🌍 Zero Waste", zero_waste_url, use_container_width=True)

def check_badges():
    """Check and award new badges"""
    stats = st.session_state.game_stats
    new_badges = []
    
    badge_conditions = [
        ('first_scan', stats['items_recycled'] >= 1),
        ('recycler_10', stats['items_recycled'] >= 10),
        ('recycler_50', stats['items_recycled'] >= 50),
        ('recycler_100', stats['items_recycled'] >= 100),
        ('streak_3', stats['current_streak'] >= 3),
        ('streak_7', stats['current_streak'] >= 7),
        ('streak_30', stats['current_streak'] >= 30),
        ('plastic_hero', stats['category_counts']['plastic'] >= 20),
        ('paper_saver', stats['category_counts']['paper'] >= 20),
        ('e_waste_pro', stats['category_counts']['e_waste'] >= 5),
    ]
    
    for badge_id, condition in badge_conditions:
        if condition and badge_id not in stats['badges_earned']:
            stats['badges_earned'].append(badge_id)
            new_badges.append(BADGES[badge_id])
    
    return new_badges

def show_gamification_sidebar():
    """Display gamification stats in sidebar"""
    init_gamification()
    stats = st.session_state.game_stats
    
    with st.sidebar:
        st.markdown("## 🎮 Your Eco Stats")
        
        # Level & XP Progress
        level, xp_in_level, xp_to_next = get_level_info(stats['total_points'])
        st.markdown(f"### Level {level} Recycler")
        
        progress = xp_in_level / xp_to_next
        st.progress(progress)
        st.caption(f"{xp_in_level}/{xp_to_next} XP to Level {level + 1}")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Points", stats['total_points'])
            st.metric("🔥 Streak", f"{stats['current_streak']} days")
        with col2:
            st.metric("♻️ Items", stats['items_recycled'])
            st.metric("🏅 Badges", len(stats['badges_earned']))
        
        # Badges
        st.markdown("---")
        st.markdown("### 🏅 Badges")
        
        if stats['badges_earned']:
            badge_icons = " ".join([BADGES[b]['icon'] for b in stats['badges_earned']])
            st.markdown(f"<div style='font-size: 24px;'>{badge_icons}</div>", unsafe_allow_html=True)
            
            with st.expander("View all badges"):
                for badge_id in stats['badges_earned']:
                    badge = BADGES[badge_id]
                    st.markdown(f"**{badge['icon']} {badge['name']}**")
                    st.caption(badge['desc'])
        else:
            st.caption("Start recycling to earn badges!")
        
        # Category breakdown
        st.markdown("---")
        st.markdown("### 📊 Recycling Breakdown")
        
        categories = stats['category_counts']
        if sum(categories.values()) > 0:
            cat_display = {
                'plastic': '🥤 Plastic',
                'paper': '📄 Paper', 
                'metal': '🥫 Metal',
                'glass': '🫙 Glass',
                'organic': '🍎 Organic',
                'e_waste': '💻 E-Waste',
                'other': '📦 Other'
            }
            for cat, count in categories.items():
                if count > 0:
                    st.markdown(f"{cat_display[cat]}: **{count}**")
        else:
            st.caption("No items recycled yet")
        
        # Leaderboard placeholder
        st.markdown("---")
        if st.button("🔄 Reset Stats", key="reset_stats"):
            st.session_state.game_stats = None
            init_gamification()
            st.rerun()

def show_achievement_popup(badges):
    """Show achievement popup for new badges"""
    for badge in badges:
        st.balloons()
        st.success(f"🎉 **Achievement Unlocked!** {badge['icon']} {badge['name']} - {badge['desc']}")

# ============== END GAMIFICATION ==============

# Function to load the recycling data
def load_recycling_data():
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'data.json')

    try:
        with open(file_path, ) as file:

            return json.load(file)
    except FileNotFoundError:
        st.error("Could not find data.json file")
        return None

def init_tool_house():
    api_key = st.secrets['api_keys']['toolhouse']
    return Toolhouse(api_key=api_key,
provider="openai")


# Initialize Groq client
def init_groq():
    api_key = st.secrets['api_keys']['groq']
    if not api_key:
        st.error("Please set your Groq API key")
        return None
    return Groq(api_key=api_key)

def init_gemini():
    api_key = st.secrets['api_keys']['gemini']
    if not api_key:
        st.error("Please set your gemini API key")
        return None
    genai.configure(api_key=api_key)

    return genai.GenerativeModel('gemini-2.0-flash')

def analyze_image_with_groq(groq_client, image_data):
    """Fallback image analysis using Groq's vision model"""
    if not groq_client:
        return "Error: Groq client not initialized"
    
    try:
        base64_image = base64.b64encode(image_data.getvalue()).decode('utf-8')
        
        prompt = """Analyze this image and identify items.
Guidelines for identification:
- List only physical objects you can clearly see
- Use simple, generic terms
- Specify quantities if multiple similar items exist
- Ignore background elements or non-disposable items
Format your response as a comma-separated list. Example:
"glass wine bottle, plastic yogurt container, banana peel, cardboard box"
"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image with Groq: {str(e)}"

def analyze_image(client, image_data, recycling_data, groq_client=None):
    if not client:
        # Try Groq directly if no Gemini
        if groq_client:
            return analyze_image_with_groq(groq_client, image_data)
        return "Error: No image analysis client available"
    
    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        # Define the prompt
        prompt = """Analyze this image and identify items.
Guidelines for identification:
- List only physical objects you can clearly see
- Use simple, generic terms
- Specify quantities if multiple similar items exist
- Ignore background elements or non-disposable items
Format your response as a comma-separated list. Example:
"glass wine bottle, plastic yogurt container, banana peel, cardboard box"
"""

        # Prepare the content for Gemini - use list format
        from PIL import Image
        import io
        
        # Reset image data position
        image_data.seek(0)
        img = Image.open(image_data)
        
        # Generate the response using Gemini with PIL image
        response = client.generate_content([prompt, img])

        # Return the generated text
        return response.text

    except Exception as e:
        # Fallback to Groq if Gemini fails
        if groq_client:
            st.warning("Gemini API unavailable, using Groq as fallback...")
            image_data.seek(0)  # Reset position
            return analyze_image_with_groq(groq_client, image_data)
        return f"Error analyzing image: {str(e)}"

def get_groq_response(client, content, prompt, th=None, location=None):
    if not client:
        return "Error: Groq client not initialized"
    
    try:
        MODEL = "llama-3.3-70b-versatile"
        messages = [
            {
                "role": "system",
                "content": content
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        
        if th is None:
            # For recycling advice - Try Gemini FIRST (more reliable)
            try:
                gemini_client = init_gemini()
                if gemini_client and prompt:
                    gemini_prompt = f"""You are a recycling expert. Based on these recycling guidelines:

{content}

Provide recycling instructions for:
{prompt}

Give clear, specific instructions for each item on how to dispose of it properly."""
                    
                    response = gemini_client.generate_content(gemini_prompt)
                    return response.text
            except Exception as gemini_err:
                pass  # Fall through to Groq
            
            # Fallback to Groq
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2000,
                )
                return response.choices[0].message.content
            except Exception as groq_err:
                error_str = str(groq_err)
                if "429" in error_str or "rate" in error_str.lower():
                    # Provide offline recycling tips based on detected items
                    return f"""⚠️ **API Rate Limit - Using Offline Guide**

Based on the items detected, here are general recycling guidelines:

**🔵 Recyclable Items (Blue Bin):**
- Paper, cardboard, newspapers, magazines
- Plastic bottles (PET), containers
- Metal cans, aluminum

**🟤 Organic Waste (Brown/Green Bin):**
- Food scraps, fruit peels
- Garden waste, leaves

**🔴 Special Disposal Required:**
- Electronics → E-waste collection center
- Batteries → Battery recycling point
- Eyeglasses → Donate to optical shops or Lions Club
- Electrical items → E-waste facility

**💡 Tip:** Search "e-waste collection near me" on Google Maps for local drop-off points.

*Rate limit resets in ~17 minutes. Try again later for detailed advice.*"""
                return f"Error: {error_str}"
    
        
    
        # Search for recycling facilities using Google Places API (better coverage in India)
        search_location = location if location else "my current location"
        
        # Get coordinates from session state if available
        lat = st.session_state.get('user_lat')
        lon = st.session_state.get('user_lon')
        
        facilities_found = []
        
        if lat and lon:
            # Try Google Places Text Search API first (better data for India)
            try:
                google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
                if google_api_key:
                    # Search for multiple types of recycling-related places
                    search_queries = [
                        "recycling center",
                        "scrap dealer kabadiwala",
                        "waste management",
                        "e-waste collection"
                    ]
                    
                    for search_query in search_queries:
                        places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                        params = {
                            'query': f"{search_query} near {search_location}",
                            'location': f"{lat},{lon}",
                            'radius': 15000,
                            'key': google_api_key
                        }
                        
                        resp = requests.get(places_url, params=params, timeout=10)
                        if resp.status_code == 200:
                            places_data = resp.json()
                            for place in places_data.get('results', [])[:5]:
                                # Avoid duplicates
                                if not any(f['name'] == place.get('name') for f in facilities_found):
                                    facilities_found.append({
                                        'name': place.get('name', 'Unknown'),
                                        'type': search_query,
                                        'lat': place.get('geometry', {}).get('location', {}).get('lat'),
                                        'lon': place.get('geometry', {}).get('location', {}).get('lng'),
                                        'address': place.get('formatted_address', ''),
                                        'phone': '',
                                        'rating': place.get('rating', ''),
                                        'open_now': place.get('opening_hours', {}).get('open_now', None)
                                    })
            except Exception as google_error:
                pass  # Fall through to OSM
            
            # Fallback to OpenStreetMap if Google didn't work
            if not facilities_found:
                try:
                    overpass_url = "https://overpass-api.de/api/interpreter"
                    
                    # Expanded search - 20km radius, more types
                    query = f"""
                    [out:json][timeout:30];
                    (
                      node["amenity"="recycling"](around:20000,{lat},{lon});
                      node["amenity"="waste_disposal"](around:20000,{lat},{lon});
                      node["amenity"="waste_transfer_station"](around:20000,{lat},{lon});
                      node["shop"="scrap"](around:20000,{lat},{lon});
                      node["shop"="second_hand"](around:20000,{lat},{lon});
                      node["industrial"="scrap_yard"](around:20000,{lat},{lon});
                      node["craft"="scrap_metal"](around:20000,{lat},{lon});
                      way["amenity"="recycling"](around:20000,{lat},{lon});
                      way["landuse"="landfill"](around:20000,{lat},{lon});
                      way["landuse"="industrial"]["name"~"waste|recycl|scrap",i](around:20000,{lat},{lon});
                    );
                    out body center;
                    """
                    
                    response = requests.get(overpass_url, params={'data': query}, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        elements = data.get('elements', [])
                        
                        for elem in elements[:15]:
                            tags = elem.get('tags', {})
                            name = tags.get('name', tags.get('operator', 'Recycling Point'))
                            e_lat = elem.get('lat') or elem.get('center', {}).get('lat')
                            e_lon = elem.get('lon') or elem.get('center', {}).get('lon')
                            
                            facilities_found.append({
                                'name': name,
                                'type': tags.get('amenity', tags.get('shop', 'facility')),
                                'lat': e_lat,
                                'lon': e_lon,
                                'address': tags.get('addr:full', tags.get('addr:street', '')),
                                'phone': tags.get('phone', tags.get('contact:phone', '')),
                                'rating': '',
                                'open_now': None
                            })
                except Exception as osm_error:
                    pass
        
        # Format results if we found facilities
        if facilities_found:
            result = f"## ♻️ Nearby Recycling Facilities\n\n"
            result += f"📍 **Found {len(facilities_found)} places near your location**\n\n"
            
            for i, f in enumerate(facilities_found, 1):
                result += f"### {i}. {f['name']}\n"
                result += f"- **Type**: {f['type'].replace('_', ' ').title()}\n"
                if f.get('address'):
                    result += f"- **Address**: {f['address']}\n"
                if f.get('phone'):
                    result += f"- **Phone**: {f['phone']}\n"
                if f.get('rating'):
                    result += f"- **Rating**: ⭐ {f['rating']}/5\n"
                if f.get('open_now') is not None:
                    status = "🟢 Open Now" if f['open_now'] else "🔴 Closed"
                    result += f"- **Status**: {status}\n"
                if f.get('lat') and f.get('lon'):
                    maps_link = f"https://www.google.com/maps/dir/?api=1&destination={f['lat']},{f['lon']}"
                    result += f"- 🗺️ [Get Directions]({maps_link})\n"
                result += "\n"
            
            return result
        
        # Fallback: No data found - provide direct Google Maps links
        gmaps_recycling = f"https://www.google.com/maps/search/recycling+center/@{lat},{lon},14z"
        gmaps_scrap = f"https://www.google.com/maps/search/scrap+dealer+kabadiwala/@{lat},{lon},14z"
        gmaps_ewaste = f"https://www.google.com/maps/search/e-waste+collection/@{lat},{lon},14z"
        
        return f"""## 🔍 Recycling Facilities near {search_location}

No facilities found in our database for this area. Click the links below to search directly on Google Maps:

### 🗺️ Search on Google Maps:

1. **[♻️ Recycling Centers]({gmaps_recycling})**
   
2. **[🛒 Scrap Dealers / Kabadiwala]({gmaps_scrap})**
   
3. **[💻 E-Waste Collection]({gmaps_ewaste})**

---

### 📞 Local Resources:

- **Dehradun Municipal Corporation**: 0135-2712055
- **Swachh Bharat Helpline**: 1969
- **Search on Justdial**: [Scrap Dealers in Dehradun](https://www.justdial.com/Dehradun/Scrap-Dealers)

📍 Your coordinates: `{lat}, {lon}`"""

    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_bin_image(waste_type):
    """Return the image path for a given waste type"""
    bin_images = {
       "battery_symbol": "images/battery_symbol.png",
       "blue": "images/blu.png",
       "brown": "images/brown.png",
       "green": "images/green.png",
       "yellow": "images/yellow.png",
       "grey": "images/grey.png",
       "oil_symbol": "images/oil_symbol.png",
       "red": "images/red.png",
       "famacie": "images/farmacie.jpg",
       "yellow_street": "images/yellow_street.png",

    }
    return bin_images.get(waste_type.lower(), None)

# Streamlit UI
def main():
    st.set_page_config(page_title="♻️ Recycling Assistant", page_icon="♻️", layout="wide")
    
    # Initialize gamification
    init_gamification()
    show_gamification_sidebar()
    
    st.title("🌍 Recycling Assistant")
    # st.write("Ask questions about how to properly sort and recycle different items")

    # tab1, tab2 = st.tabs(["📸 Capture Image", "📝 Analysis Results"])
    
    # # Load recycling data
    recycling_data = load_recycling_data()

    if not recycling_data:
        st.stop()

    th = init_tool_house()

    groq_client = init_groq()
    openai_client = init_gemini()
  

    if not groq_client or not openai_client or not th:
        st.stop()


    img_file = st.camera_input("Take a picture of the item")
    if img_file is not None:
        with st.spinner("Analyzing image..."):
            identified_items = analyze_image(openai_client, img_file, recycling_data, groq_client)
            
            if not isinstance(identified_items, str) or identified_items.startswith("Error"):
                st.error(identified_items)
            else:
                st.write("**Detected Items:**", identified_items)

                items = identified_items
                context = json.dumps(recycling_data)
                GROQ_CONTENT = """You are a specialized recycling assistant with deep knowledge of waste sorting.
                    Your goal is to provide accurate, practical advice that helps users correctly dispose of items.
                    Always prioritize environmental safety and proper waste separation."""
                GROQ_PROMPT = f"""You are a recycling expert assistant. Using the provided recycling guidelines, analyze these items: {items} Context (recycling guidelines):
{context}
For each item, provide a structured analysis:
1. Item Name:
- Correct Bin: [Specify the exact bin color/type]
- Preparation Required: [List any cleaning/preparation steps]
- Reason: [Explain why this bin is correct]
- Special Notes: [Any warnings, alternatives, or important details]
Guidelines for your response:
- Separate each item with a blank line
- Be specific about bin colors and types
- If an item isn't in the guidelines, recommend the safest disposal method
- Mention if items need to be clean, disassembled, or specially prepared
- Include any relevant warnings about contamination or hazardous materials
- If an item has multiple components, explain how to separate them
Please format your response clearly and concisely for each item."""

                recycling_advice = get_groq_response(groq_client, GROQ_CONTENT, GROQ_PROMPT)
                
                st.write("### Recycling Instructions:")
                advice_items = recycling_advice.split('\n\n')
                
                for item in advice_items:
                    if item.strip():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(item)
                        with col2:
                            # Map waste types to their corresponding images
                            if "unsorted waste" in item.lower() or "grey" in item.lower():
                                st.image(get_bin_image("grey"), width=200)
                            elif "organic" in item.lower() or "food waste" in item.lower() or "brown" in item.lower():
                                st.image(get_bin_image("brown"), width=200)
                            elif "plastic" in item.lower() or "metal" in item.lower() or "yellow" in item.lower():
                                st.image(get_bin_image("yellow"), width=200)
                            elif "paper" in item.lower() or "blue" in item.lower():
                                st.image(get_bin_image("blue"), width=200)
                            elif "collection centers" in item.lower() or "electronic" in item.lower() or "red" in item.lower():
                                st.image(get_bin_image("red"), width=200)
                            elif "oil" in item.lower():
                                st.image(get_bin_image("oil_symbol"), width=200)
                            elif "battery" in item.lower():
                                st.image(get_bin_image("battery_symbol"), width=200),
                            elif "farmacy" in item.lower():
                                st.image(get_bin_image("farmacie"), width=200)
                
                # 🎮 GAMIFICATION: Award points directly for scanning
                items = [i.strip() for i in identified_items.split(',') if i.strip()]
                points_earned = len(items) * 10  # 10 points per item
                
                init_gamification()
                stats = st.session_state.game_stats
                stats['total_points'] += points_earned
                stats['items_recycled'] += len(items)
                
                # Update streak
                today = datetime.now().date()
                if stats['last_scan_date']:
                    last_date = datetime.strptime(stats['last_scan_date'], '%Y-%m-%d').date()
                    if today == last_date:
                        pass  # Same day, no change
                    elif today - last_date == timedelta(days=1):
                        stats['current_streak'] += 1
                    else:
                        stats['current_streak'] = 1
                else:
                    stats['current_streak'] = 1
                
                stats['last_scan_date'] = today.strftime('%Y-%m-%d')
                stats['longest_streak'] = max(stats['longest_streak'], stats['current_streak'])
                stats['level'], _, stats['xp_to_next_level'] = get_level_info(stats['total_points'])
                
                # Check for badges
                new_badges = check_badges()
                
                # Show points earned
                st.success(f"🎮 **+{points_earned} points earned!** ({len(items)} items scanned)")
                if new_badges:
                    for badge in new_badges:
                        st.info(f"🏅 **Badge Unlocked:** {badge['icon']} {badge['name']} - {badge['desc']}")
                
                # 🎬 Show YouTube video suggestions for recyclable items
                display_youtube_suggestions(items)
    
    # Add the ecological sites finder with automatic location
    st.markdown("---")
    st.subheader("🔍 Find Nearby Ecological Sites")
    
    # Initialize session state
    if 'detected_location' not in st.session_state:
        st.session_state.detected_location = None
        st.session_state.gps_coords = None
        st.session_state.auto_search = False
    
    # Check URL for GPS coordinates (from JavaScript redirect)
    query_params = st.query_params
    if 'lat' in query_params and 'lon' in query_params:
        lat = float(query_params['lat'])
        lon = float(query_params['lon'])
        
        if st.session_state.gps_coords != (lat, lon):
            st.session_state.gps_coords = (lat, lon)
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon
            try:
                geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
                headers = {'User-Agent': 'RecyclingAssistant/1.0'}
                geo_response = http_requests.get(geocode_url, headers=headers, timeout=5)
                geo_data = geo_response.json()
                st.session_state.detected_location = geo_data.get('display_name', f"{lat}, {lon}")
            except:
                st.session_state.detected_location = f"GPS: {lat:.6f}, {lon:.6f}"
    
    # Single click button - Get location and search
    if st.button("📍 Find Recycling Facilities Near Me", type="primary", key="one_click_search"):
        st.session_state.auto_search = True
        st.rerun()
    
    # Auto search flow
    if st.session_state.auto_search:
        # Try to get geolocation
        location = get_geolocation()
        
        if location and 'coords' in location:
            lat = location['coords']['latitude']
            lon = location['coords']['longitude']
            acc = location['coords'].get('accuracy', 'unknown')
            
            # Reverse geocode
            try:
                geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18"
                headers = {'User-Agent': 'RecyclingAssistant/1.0'}
                geo_response = http_requests.get(geocode_url, headers=headers, timeout=5)
                geo_data = geo_response.json()
                detected_loc = geo_data.get('display_name', f"{lat}, {lon}")
            except:
                detected_loc = f"GPS: {lat:.6f}, {lon:.6f}"
            
            st.session_state.detected_location = detected_loc
            st.session_state.gps_coords = (lat, lon)
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon
            
            st.success(f"📍 **Your Location:** {detected_loc} (Accuracy: {acc}m)")
            
            # Auto search immediately
            with st.spinner("🔍 Searching for recycling facilities near you..."):
                ecological_sites = get_groq_response(groq_client, '', '', th, location=detected_loc)
                st.write("### 📍 Nearby Ecological Sites:")
                st.write(ecological_sites)
                
                # 🎮 GAMIFICATION: Award badge for finding facilities
                if 'location_finder' not in st.session_state.game_stats['badges_earned']:
                    st.session_state.game_stats['badges_earned'].append('location_finder')
                    st.session_state.game_stats['total_points'] += 25
                    st.balloons()
                    st.success("🎉 **Achievement Unlocked!** 🗺️ Explorer - Found recycling facilities! (+25 points)")
            
            st.session_state.auto_search = False
            
            if st.button("🔄 Search Again", key="search_again"):
                st.session_state.auto_search = True
                st.rerun()
        else:
            st.info("⏳ Getting your location... Please allow location access in browser.")
            st.warning("If prompted, click **Allow** for location access, then click the button below.")
            if st.button("🔄 Continue After Allowing Location", key="continue_after_allow"):
                st.rerun()
    
    # Show previous location if available (when not in auto search mode)
    elif st.session_state.detected_location:
        st.success(f"📍 **Last Location:** {st.session_state.detected_location}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Search This Location", key="search_prev"):
                with st.spinner("Searching for recycling facilities..."):
                    ecological_sites = get_groq_response(groq_client, '', '', th, location=st.session_state.detected_location)
                    st.write("### 📍 Nearby Ecological Sites:")
                    st.write(ecological_sites)
        with col2:
            if st.button("🔄 Clear", key="clear_loc"):
                st.session_state.detected_location = None
                st.session_state.gps_coords = None
                st.query_params.clear()
                st.rerun()
    
    # Manual input fallback
    st.markdown("---")
    st.caption("Or enter location manually:")
    user_location = st.text_input("Location", placeholder="e.g., Delhi, India", key="manual_location")
    if user_location and st.button("Search", key="manual_search"):
        with st.spinner(f"Searching near {user_location}..."):
            ecological_sites = get_groq_response(groq_client, '', '', th, location=user_location)
            st.write("### 📍 Nearby Ecological Sites:")
            st.write(ecological_sites)
            
if __name__ == "__main__":
    main()
