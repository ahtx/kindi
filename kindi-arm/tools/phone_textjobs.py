#!/usr/bin/env python3
import json, os, sys, time, traceback, re
import numpy as np

# data copied from kbench.textjobs_data (158 items)
SENTIMENT = [
    ("The battery lasts two full days and the screen is bright. Very happy with it.", "positive"),
    ("Stopped working after a week. Support never answered my emails.", "negative"),
    ("The package arrived on Tuesday.", "neutral"),
    ("Best coffee grinder I have owned. Quiet and consistent.", "positive"),
    ("The zipper broke the first time I used the bag.", "negative"),
    ("It comes in three colors: black, blue and gray.", "neutral"),
    ("Setup took five minutes and everything just worked.", "positive"),
    ("Way too expensive for what you get. I returned it.", "negative"),
    ("The manual is printed in English and Spanish.", "neutral"),
    ("My kids love it and use it every day.", "positive"),
    ("The app crashes every time I open the settings page.", "negative"),
    ("The product weighs 1.2 kilograms.", "neutral"),
    ("Excellent build quality, and the price was fair.", "positive"),
    ("Terrible smell from the plastic that does not go away.", "negative"),
    ("It fits inside a standard backpack pocket.", "neutral"),
    ("The sound is clear and the bass is strong for the size.", "positive"),
    ("Arrived scratched and the box was crushed.", "negative"),
    ("Replacement filters are sold separately.", "neutral"),
    ("Customer service solved my problem in one call.", "positive"),
    ("The battery drains in under two hours. Very disappointed.", "negative"),
    ("The device connects over Bluetooth or a cable.", "neutral"),
    ("I have recommended this to all my friends.", "positive"),
    ("Instructions were missing and I could not assemble it.", "negative"),
    ("The handle is made of stainless steel.", "neutral"),
    ("Light, fast and exactly as described.", "positive"),
    ("Keeps disconnecting and I have to pair it again.", "negative"),
    ("It includes a one-meter charging cable.", "neutral"),
    ("The picture quality is outstanding for the price.", "positive"),
    ("Buttons stopped responding after a month.", "negative"),
    ("The warranty covers two years.", "neutral"),
]
ESCALATE = [
    ("My child swallowed part of this toy and is choking. What do I do?", "yes"),
    ("How do I change the shipping address on my order?", "no"),
    ("Someone used my credit card to place orders I did not make.", "yes"),
    ("Can I get this jacket in a larger size?", "no"),
    ("There is smoke coming from the charger and it smells like burning.", "yes"),
    ("I would like to leave a review for the blender I bought.", "no"),
    ("I received a message saying your company will sue me if I do not pay today.", "yes"),
    ("What time do you open on Saturdays?", "no"),
    ("The whole neighborhood has no power and your website is down too.", "yes"),
    ("Do you have this model in blue?", "no"),
    ("I think my account was hacked and my personal data was leaked.", "yes"),
    ("Could you recommend a good case for this tablet?", "no"),
    ("I am feeling very hopeless and I do not want to be here anymore.", "yes"),
    ("The color is slightly different from the photo, but it is fine.", "no"),
    ("Your driver hit my parked car and left the scene.", "yes"),
    ("I forgot my password. How can I reset it?", "no"),
    ("I found a piece of glass inside the jar of baby food.", "yes"),
    ("When will the black version be back in stock?", "no"),
    ("My pacemaker started beeping after I used your device near it.", "yes"),
    ("Can you send me a copy of my receipt?", "no"),
    ("Someone is pretending to be your company and asking for my bank details.", "yes"),
    ("I would prefer a refund to store credit, if possible.", "no"),
    ("The medicine I received has a different name on the label than what I ordered.", "yes"),
    ("Does this work with Windows 11?", "no"),
    ("There is a gas leak smell coming from the heater you installed.", "yes"),
    ("The package is a day late but the tracking shows it is coming.", "no"),
    ("I need to report that an employee threatened me with violence.", "yes"),
    ("Can I add a gift message to the order?", "no"),
    ("My bank says there are several charges from you that I did not authorize.", "yes"),
    ("Is it safe to put this pan in the dishwasher?", "no"),
]
SPAM = [
    ("Congratulations! You have won a $1,000 gift card. Click here to claim now.", "spam"),
    ("Hi, are we still meeting at 3 pm tomorrow in room 204?", "ham"),
    ("URGENT: Your account will be closed. Verify your password at this link.", "spam"),
    ("Please find attached the minutes from today's meeting.", "ham"),
    ("Make money fast from home!!! No experience needed, start today.", "spam"),
    ("The report you asked for is ready. Let me know if you need changes.", "ham"),
    ("You have been selected for a free cruise. Call now to confirm.", "spam"),
    ("Can you review the draft before Friday?", "ham"),
    ("Lose 10 kg in one week with this one strange pill.", "spam"),
    ("Lunch was great yesterday, thanks for the invitation.", "ham"),
    ("Dear user, we detected unusual activity. Enter your card number here.", "spam"),
    ("The invoice for March is attached. Payment is due in 30 days.", "ham"),
    ("Earn $5000 per week!!! Limited spots, register now.", "spam"),
    ("I'll pick up the kids at 5. See you at home.", "ham"),
    ("Hot singles in your area want to meet you tonight.", "spam"),
    ("Here are the photos from the weekend trip.", "ham"),
    ("Your package could not be delivered. Open this file to reschedule.", "spam"),
    ("Thanks for your help with the presentation, it went well.", "ham"),
    ("Act now: lowest mortgage rates, guaranteed approval, no credit check.", "spam"),
    ("Could we move our 10 am call to 11 am?", "ham"),
    ("You are our 1,000,000th visitor! Claim your prize immediately.", "spam"),
    ("The train leaves at 8:15 from platform 3.", "ham"),
    ("Double your Bitcoin in 24 hours. Send coins to this address.", "spam"),
    ("Happy birthday! Hope you have a wonderful day.", "ham"),
    ("Final notice: your tax refund is waiting. Submit your details.", "spam"),
    ("I have booked the conference room for Thursday afternoon.", "ham"),
    ("Exclusive offer: luxury watches at 90% off, today only.", "spam"),
    ("The document has been signed and returned.", "ham"),
    ("We tried to reach you about your car's extended warranty.", "spam"),
    ("Mom called and asked you to bring milk on the way home.", "ham"),
]
TOPIC = [
    ("The home team scored in the final minute to win the championship.", "sports"),
    ("The senate passed the new tax bill after a long debate.", "politics"),
    ("The company reported record quarterly profits and raised its forecast.", "business"),
    ("The new chip uses less power and runs artificial intelligence models faster.", "technology"),
    ("Doctors recommend at least thirty minutes of exercise each day.", "health"),
    ("The striker signed a three-year contract with the club.", "sports"),
    ("The mayor announced plans to run for governor next year.", "politics"),
    ("Shares fell after the central bank raised interest rates.", "business"),
    ("Researchers released an update that fixes a major security flaw in the browser.", "technology"),
    ("A new study links poor sleep to higher blood pressure.", "health"),
    ("The marathon will close several streets in the city center on Sunday.", "sports"),
    ("Voters will choose a new representative in next month's election.", "politics"),
    ("The airline announced new routes and lower fares for the summer.", "business"),
    ("The phone maker unveiled a foldable screen at its annual event.", "technology"),
    ("Health officials urged residents to get vaccinated before the winter season.", "health"),
    ("The goalkeeper made three saves in the penalty shootout.", "sports"),
    ("Parliament voted to increase funding for public schools.", "politics"),
    ("The startup raised $50 million to expand into new markets.", "business"),
    ("Scientists trained a model that can predict the weather more accurately.", "technology"),
    ("Eating more vegetables and whole grains can lower the risk of heart disease.", "health"),
    ("The coach resigned after the team lost six games in a row.", "sports"),
    ("The two countries signed a trade agreement after months of talks.", "politics"),
    ("Oil prices rose as demand increased in Asia.", "business"),
    ("The company launched a satellite to provide internet access in rural areas.", "technology"),
    ("The hospital opened a new clinic for children's mental health.", "health"),
    ("She won the gold medal in the 100-meter final.", "sports"),
    ("The president vetoed the bill and sent it back to congress.", "politics"),
    ("The merger will create the largest retailer in the region.", "business"),
    ("Developers released a tool that helps find bugs in code automatically.", "technology"),
    ("Regular hand washing reduces the spread of infections.", "health"),
]
SPELL = [
    ("I recieved the packege yesteday.", "I received the package yesterday."),
    ("Please let me no if you have any questons.", "Please let me know if you have any questions."),
    ("The resturant was very croweded on friday.", "The restaurant was very crowded on Friday."),
    ("She is definately comming to the meeeting.", "She is definitely coming to the meeting."),
    ("We need to adress this problme immediatly.", "We need to address this problem immediately."),
    ("He bougth a new bycicle last weak.", "He bought a new bicycle last week."),
    ("The goverment announced a new polisy.", "The government announced a new policy."),
    ("I beleive this is the corect anser.", "I believe this is the correct answer."),
    ("Thier house is bigger then ours.", "Their house is bigger than ours."),
    ("The calender shows the wrong mounth.", "The calendar shows the wrong month."),
    ("It is neccessary to bring your identifcation.", "It is necessary to bring your identification."),
    ("The libary closes at six oclock.", "The library closes at six o'clock."),
    ("She wrote a beautifull leter to her freind.", "She wrote a beautiful letter to her friend."),
    ("The wheather will be sunnier tommorow.", "The weather will be sunnier tomorrow."),
    ("I finaly finished the assigment on time.", "I finally finished the assignment on time."),
]
GRAMMAR = [
    ("She don't like coffee in the morning.", "She doesn't like coffee in the morning."),
    ("He go to school every day by bus.", "He goes to school every day by bus."),
    ("There is many reasons to visit the city.", "There are many reasons to visit the city."),
    ("I have seen him yesterday at the store.", "I saw him yesterday at the store."),
    ("She is more taller than her brother.", "She is taller than her brother."),
    ("Me and him went to the store.", "He and I went to the store."),
    ("The book was wrote by a famous author.", "The book was written by a famous author."),
    ("If I was you, I would take the job.", "If I were you, I would take the job."),
    ("He runned very fast to catch the train.", "He ran very fast to catch the train."),
    ("Each of the students have a laptop.", "Each of the students has a laptop."),
    ("She can sings very well.", "She can sing very well."),
    ("I am interesting in learning new languages.", "I am interested in learning new languages."),
]
TABLES = [
    ("Name: Alice, Age: 30, City: Oslo. Name: Bob, Age: 25, City: Bergen.", ["Name", "Age", "City"], 2),
    ("Product: Pen, Price: 2, Stock: 50. Product: Notebook, Price: 5, Stock: 20. Product: Eraser, Price: 1, Stock: 100.", ["Product", "Price", "Stock"], 3),
    ("Day: Monday, Task: Report, Done: yes. Day: Tuesday, Task: Meeting, Done: no. Day: Wednesday, Task: Review, Done: yes.", ["Day", "Task", "Done"], 3),
    ("Country: Norway, Capital: Oslo. Country: Sweden, Capital: Stockholm. Country: Denmark, Capital: Copenhagen.", ["Country", "Capital"], 2),
    ("Title: Dune, Author: Herbert, Year: 1965. Title: 1984, Author: Orwell, Year: 1949.", ["Title", "Author", "Year"], 2),
]
NOTES = [
    "buy milk - call dentist asap - finish report by friday - pick up dry cleaning",
    "schedule team meeting - urgent: fix login bug - update documentation - reply to anna",
    "pay electricity bill today - book flight - water plants - prepare slides for monday",
]
NEWS = [
    "The city council approved a plan on Tuesday to build a new bridge across the river. Construction is expected to begin next spring and take two years to complete.",
    "A local school opened a new library this week with more than 10,000 books and a reading room for students. The project was funded by community donations.",
    "Researchers at the national university announced that they have developed a battery that charges in five minutes. The technology could be used in phones and electric cars within a few years.",
]

CLASSIFY = {
    "sentiment": (SENTIMENT, ["positive", "negative", "neutral"],
                  "Classify the sentiment of this customer review as positive, negative or neutral. A neutral review states facts with no opinion."),
    "escalate": (ESCALATE, ["yes", "no"],
                 "You triage customer messages. Answer yes if a person must look at this message now (danger to health or safety, fraud, a legal threat, a security or privacy breach, a major outage, or a person in crisis). Answer no for routine questions, requests and feedback."),
    "spam": (SPAM, ["spam", "ham"],
             "Is this e-mail spam (unwanted advertising, a scam or phishing) or ham (a normal message)? Answer spam or ham."),
    "topic": (TOPIC, ["sports", "politics", "business", "technology", "health"],
              "Give the topic of this news sentence: sports, politics, business, technology or health."),
}
JOBS = list(CLASSIFY) + ["spell", "grammar", "table", "notes", "news"]

def norm(s):
    s = s.strip().strip('"').strip()
    for a, b in (("doesn't", "does not"), ("didn't", "did not"), ("don't", "do not"), ("isn't", "is not"),
                 ("aren't", "are not"), ("wasn't", "was not"), ("weren't", "were not")):
        s = re.sub(r"\b%s\b" % a, b, s, flags=re.I)
    return re.sub(r"\s+", " ", s).rstrip(".").lower()

def jobs():
    out = []
    for name, (items, labels, instr) in CLASSIFY.items():
        for text, want in items:
            out.append({"job": name, "input": text, "want": want, "labels": labels, "max_tokens": 8,
                        "prompt": "%s Reply with one word only.\n\n%s" % (instr, text)})
    for name, items, instr in (("spell", SPELL, "Correct the spelling mistakes in this sentence. Change nothing else. Reply with only the corrected sentence."),
                               ("grammar", GRAMMAR, "Correct the grammar of this sentence. Change as little as possible. Reply with only the corrected sentence.")):
        for text, want in items:
            out.append({"job": name, "input": text, "want": want, "max_tokens": 60, "prompt": "%s\n\n%s" % (instr, text)})
    for text, header, rows in TABLES:
        out.append({"job": "table", "input": text, "want": {"header": header, "rows": rows}, "max_tokens": 200,
                    "prompt": "Turn this text into a Markdown table with the columns: %s. Reply with only the table.\n\n%s" % (", ".join(header), text)})
    for text in NOTES:
        out.append({"job": "notes", "input": text, "want": None, "max_tokens": 120,
                    "prompt": "Clean up these notes. Write them as a short list of clear tasks, one per line, with the most urgent task first. Keep every fact.\n\n" + text})
    for text in NEWS:
        out.append({"job": "news", "input": text, "want": None, "max_tokens": 80,
                    "prompt": "Write a headline of at most ten words and a one-sentence summary for this news story. Format: Headline: ... / Summary: ...\n\n" + text})
    return out

def score(j, got):
    g = got.strip()
    if "labels" in j:
        w = re.findall(r"[a-z]+", g.lower())
        pick = next((x for x in w if x in j["labels"]), None)
        return pick == j["want"], pick
    if j["job"] in ("spell", "grammar"):
        wants = [j["want"]] + (["Him and me went to the store.", "He and I went to the store."] if "Me and him" in j["input"] else [])
        line = g.splitlines()[0] if g else ""
        return any(norm(line) == norm(w) for w in wants), line
    if j["job"] == "table":
        rows = [r for r in g.splitlines() if r.strip().startswith("|")]
        body = [r for r in rows[2:] if r.strip("| -:")]
        head = [c.strip().lower() for c in rows[0].strip().strip("|").split("|")] if rows else []
        ok = len(rows) >= 3 and head == [h.lower() for h in j["want"]["header"]] and len(body) == j["want"]["rows"]
        return ok, "%d rows" % len(body)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='max items per job (0=all)')
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    from kindi.cpu import Engine
    from kindi.tokenizer import Tok
    from kindi.chat import render, template_variant
    eng = Engine(args.pack, ctx=args.ctx, threads=args.threads, log=lambda m: print(m,flush=True))
    vocab = json.load(open(os.path.join(args.pack,'vocab.json')))
    tok = Tok(vocab, os.path.join(args.pack,'merges.txt'))
    variant = template_variant(args.pack)

    def gen(user, max_new):
        msgs=[{'role':'user','content':user}]
        body,gp = render(msgs, tools=None, enable_thinking=False, variant=variant)
        ids = tok.encode(body+gp)
        eng.reset()
        t0=time.perf_counter()
        for i,t in enumerate(ids[:-1]): eng.feed(t,i)
        out=[eng.step(ids[-1], len(ids)-1)]; pos=len(ids)
        for _ in range(max_new-1):
            if out[-1] in (151645,151643,151644): break
            nt=eng.step(out[-1],pos); out.append(nt); pos+=1
            if nt in (151645,151643,151644): break
        t1=time.perf_counter()
        gen_ids=list(out)
        while gen_ids and gen_ids[-1] in (151645,151643,151644): gen_ids.pop()
        return tok.decode(gen_ids), t1-t0, len(gen_ids)

    only=[x for x in args.only.split(',') if x]
    js = jobs()
    if only: js=[j for j in js if j['job'] in only]
    if args.limit:
        per={}
        kept=[]
        for j in js:
            per.setdefault(j['job'],0)
            if per[j['job']] < args.limit:
                kept.append(j); per[j['job']]+=1
        js=kept
    # checkpointed/resume-safe: load existing partial results for same job list
    res=[]
    done=0
    if os.path.exists(args.out):
        try:
            prev=json.load(open(args.out))
            if isinstance(prev, dict) and isinstance(prev.get('items'), list):
                res=prev['items']
                done=len(res)
                if done>len(js):
                    res=res[:len(js)]; done=len(res)
                print('RESUME %d/%d from %s' % (done,len(js),args.out), flush=True)
        except Exception as e:
            print('RESUME ignored (%s), starting fresh' % e, flush=True)
            res=[]; done=0

    def write_checkpoint(wall_s):
        summary={}
        for r in res:
            s=summary.setdefault(r['job'],{'n':0,'ok':0,'secs':0.0})
            s['n']+=1; s['ok']+=bool(r['ok']); s['secs']+=float(r.get('secs') or 0)
        for k,s in summary.items():
            s['secs']=round(s['secs'],2); s['mean_secs']=round(s['secs']/s['n'],2) if s['n'] else 0
        payload={'wall_s':round(wall_s,1),'completed':len(res),'total':len(js),'summary':summary,'items':res}
        tmp=args.out+'.tmp'
        with open(tmp,'w') as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,args.out)

    t0=time.time()
    base_wall=0.0
    try:
        if os.path.exists(args.out):
            prev=json.load(open(args.out))
            base_wall=float(prev.get('wall_s') or 0.0)
    except Exception:
        base_wall=0.0
    for i in range(done,len(js)):
        j=js[i]
        try:
            got,secs,nt = gen(j['prompt'], j['max_tokens'])
            ok,parsed = score(j, got)
        except Exception as e:
            got,secs,nt,ok,parsed = '',0,0,False,str(e)
        r=dict(j); r.update({'output':got,'ok':ok,'parsed':parsed,'secs':round(secs,3),'gen_tokens':nt})
        res.append(r)
        wall=base_wall + (time.time()-t0)
        mark = 'OK' if ok else ('..' if ok is None else 'XX')
        print('[%d/%d] %s %s %.2fs %s' % (i+1,len(js),j['job'],mark,secs,(got.strip()[:60].replace('\n',' '))), flush=True)
        try:
            write_checkpoint(wall)
        except Exception as e:
            print('CHECKPOINT_ERR %s' % e, flush=True)
    wall=base_wall + (time.time()-t0)
    write_checkpoint(wall)
    print('DONE wall %.1fs -> %s' % (wall,args.out), flush=True)
    summary={}
    for r in res:
        s=summary.setdefault(r['job'],{'n':0,'ok':0,'secs':0.0})
        s['n']+=1; s['ok']+=bool(r['ok']); s['secs']+=r['secs']
    for k,s in summary.items():
        s['secs']=round(s['secs'],2); s['mean_secs']=round(s['secs']/s['n'],2) if s['n'] else 0
    for k,s in summary.items():
        print('  %s %s/%s' % (k,s['ok'],s['n']))

if __name__=='__main__':
    import argparse
    main()
