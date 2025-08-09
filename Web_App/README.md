
## WHAT TO EDIT FOR COMMON CHANGES

### 🎯 MAIN PAGE TEXT & BUTTONS
**FILE: `app/page.tsx`**
- Website header title: Search for "Warehouse Agents Navigation" (line ~85)
- Hero section text: Look around lines 100-120
- Button labels: Search for `<Button>` tags
- Tab names: Search for `<TabsTrigger>` (Interactive Demo, Algorithm Comparison, etc.)
- Control panel labels: Search for `<label>` tags (Episodes, Learning Rate, etc.)
- Dropdown options: Search for `<SelectItem>` tags

### 🏷️ BROWSER TAB TITLE
**FILE: `app/layout.tsx`**
- Tab title: Look for `title: "Warehouse Agents Navigation"`
- Page description: Look for `description:` field

### 🤖 ALGORITHM INFORMATION
**FILE: `components/algorithm-comparison.tsx`**
- Algorithm descriptions: Look for `algorithms` array (lines ~10-60)
- Pros/cons lists: Inside each algorithm object
- Performance table: Look for comparison table section
- "Best suited for" badges: Look for `bestFor` arrays

### 🏭 WAREHOUSE ENVIRONMENT INFO
**FILE: `components/warehouse-info.tsx`**
- Environment overview: Look for "Environment Overview" section
- Agent type descriptions: Look for "Agent Types" section  
- Rules and constraints: Look for "Operational Rules" section
- Reward structure: Look for "Reward Structure" section

### 📚 DOCUMENTATION CONTENT
**FILE: `components/documentation-modal.tsx`**
- README text: Look for `readmeContent` variable (contains all documentation)
- Modal title: Look for `DialogTitle`

### 💻 CODE EXAMPLES
**FILE: `components/code-viewer-modal.tsx`**
- Environment code: Look for `envCode` variable
- DQN code: Look for `dqnCode` variable  
- PPO code: Look for `ppoCode` variable
- Button labels: Search for "Save", "Reset", "View Code"

### 📊 TRAINING RESULTS DATA
**FILE: `app/page.tsx`** (Training Results Tab section)
- Performance numbers: Search for "Average Reward:", "Success Rate:", etc.
- Look around lines 400-500 for the results cards

## COMMON MODIFICATION EXAMPLES

### Change Main Title:
1. Open `app/layout.tsx` → Change `title: "Your New Title"`
2. Open `app/page.tsx` → Search "Warehouse Agents Navigation" → Replace both instances

### Change Hero Description:
1. Open `app/page.tsx`
2. Look for the paragraph starting with "Interactive demonstration of DQN and PPO..."
3. Replace with your text

### Add New Algorithm Info:
1. Open `components/algorithm-comparison.tsx`
2. Find the `algorithms` array
3. Add new object with same structure as existing ones

### Change Environment Rules:
1. Open `components/warehouse-info.tsx`
2. Find the section you want to modify (Agent Types, Rules, etc.)
3. Edit the text in the respective section

### Update Documentation:
1. Open `components/documentation-modal.tsx`
2. Find `readmeContent` variable
3. Replace the entire string with your new documentation

### Change Button Text:
1. Open `app/page.tsx`
2. Search for the button text you want to change
3. Look for `<Button>` tags containing that text

## SEARCH TIPS
- Use Ctrl+F (or Cmd+F) to search for specific text you want to change
- Search for exact button text like "Start Training" or "Documentation"
- Search for section titles like "Algorithm Comparison" or "Environment Details"
- Look for HTML-like tags: `<Button>`, `<CardTitle>`, `<TabsTrigger>`

## FILE PRIORITY FOR BEGINNERS
1. **`app/page.tsx`** - Most important, contains 80% of visible text
2. **`components/warehouse-info.tsx`** - Environment descriptions
3. **`components/algorithm-comparison.tsx`** - Algorithm details
4. **`components/documentation-modal.tsx`** - Documentation popup
5. **`app/layout.tsx`** - Browser tab title

## BACKUP REMINDER
Always make a copy of files before editing them!