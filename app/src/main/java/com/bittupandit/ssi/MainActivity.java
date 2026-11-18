package com.bittupandit.ssi;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.content.Intent;
import android.net.Uri;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class MainActivity extends Activity {
    private static final String REPO = "https://github.com/egalego-sudo/special-super-intelligency";
    private static final String RAW_CONFIG = "https://raw.githubusercontent.com/egalego-sudo/special-super-intelligency/android-control-app-v1/app-config/dashboard.json";
    private static final int BG = Color.rgb(246, 248, 252);
    private static final int INK = Color.rgb(25, 34, 52);
    private static final int MUTED = Color.rgb(102, 115, 136);
    private static final int ACCENT = Color.rgb(89, 73, 196);
    private static final int TEAL = Color.rgb(0, 139, 135);
    private LinearLayout root, moduleList;
    private TextView syncStatus, versionText;
    private final ArrayList<Module> modules = new ArrayList<>();

    private static class Module {
        String id, title, summary, details, status, anchor;
        Module(JSONObject j) {
            id=j.optString("id"); title=j.optString("title"); summary=j.optString("summary");
            details=j.optString("details"); status=j.optString("status"); anchor=j.optString("anchor");
        }
    }

    private int dp(float v) { return (int)(v * getResources().getDisplayMetrics().density + 0.5f); }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.rgb(246,248,252));
        getWindow().setNavigationBarColor(Color.rgb(246,248,252));
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
        buildShell();
        loadFallbackModules();
        renderModules();
        syncDashboard();
    }

    private void buildShell() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18),dp(20),dp(18),dp(28));
        root.setBackgroundColor(BG);
        scroll.addView(root);
        setContentView(scroll);

        TextView eyebrow = label("PRIVATE RESEARCH • ANDROID CONTROL",11,TEAL,true);
        root.addView(eyebrow);
        TextView title = label("SPECIAL SUPER\nINTELLIGENCY",27,INK,true);
        title.setPadding(0,dp(8),0,dp(4));
        root.addView(title);
        root.addView(label("14-MODULE CONTROL CENTER",12,MUTED,true));
        root.addView(space(16));

        LinearLayout hero = panel(Color.rgb(29,34,58),18);
        TextView heroTag=label("PROJECT COMMAND",11,Color.rgb(174,222,220),true);
        TextView heroTitle=label("One place. Fourteen workstreams.",19,Color.WHITE,true);
        TextView heroNote=label("Track research, open each module and sync its live dashboard configuration.",13,Color.rgb(222,226,239),false);
        hero.addView(heroTag); hero.addView(heroTitle); hero.addView(heroNote);
        root.addView(hero, bottom(14));

        LinearLayout actionRow = new LinearLayout(this);
        actionRow.setOrientation(LinearLayout.HORIZONTAL);
        Button sync = actionButton("⟳  SYNC DASHBOARD",ACCENT,Color.WHITE);
        sync.setOnClickListener(v -> syncDashboard());
        actionRow.addView(sync,new LinearLayout.LayoutParams(0,-2,1));
        Button repo = actionButton("REPOSITORY",Color.rgb(226,230,240),INK);
        repo.setOnClickListener(v -> open(REPO));
        LinearLayout.LayoutParams rp=new LinearLayout.LayoutParams(0,-2,1); rp.leftMargin=dp(8);
        actionRow.addView(repo,rp);
        root.addView(actionRow,bottom(10));

        syncStatus=label("Loading live dashboard configuration…",11,MUTED,false);
        root.addView(syncStatus);
        versionText=label("Dashboard config: local fallback",11,MUTED,false);
        root.addView(versionText,bottom(18));

        TextView section=label("YOUR 14 RESEARCH MODULES",14,INK,true);
        root.addView(section,bottom(10));
        moduleList=new LinearLayout(this);
        moduleList.setOrientation(LinearLayout.VERTICAL);
        root.addView(moduleList);

        LinearLayout quick = panel(Color.WHITE,16);
        quick.addView(label("ENGINEERING SHORTCUTS",11,ACCENT,true));
        quick.addView(label("Tests, progress and review history",15,INK,true));
        addWideAction(quick,"OPEN TESTS & BUILD HISTORY",REPO+"/actions");
        addWideAction(quick,"OPEN PROJECT PROGRESS",REPO+"/blob/android-control-app-v1/PROGRESS.md");
        addWideAction(quick,"OPEN 14-POINT ROADMAP",REPO+"/blob/android-control-app-v1/docs/roadmap/14-point-roadmap.md");
        root.addView(quick,bottom(14));

        TextView creator=label("Creator by Bittu Pandit",12,MUTED,true);
        creator.setGravity(Gravity.CENTER);
        root.addView(creator);
        root.addView(label("Config updates do not require an APK rebuild. Native app changes do.",10,MUTED,false));
    }

    private void loadFallbackModules() {
        String[][] rows = {
            {"independent-foundation-model","Independent Foundation Model","अपना मूल मॉडल, tokenizer और training pipeline","स्वतंत्र foundation model का निर्माण। अभी prototype पूर्ण trained model नहीं है।","Research foundation only"},
            {"extreme-reasoning-intelligence","Extreme Reasoning Intelligence","Multi-step reasoning और verification","योजना, तर्क और adversarial परीक्षण विकसित करना।","Planned"},
            {"self-evolving-ai-architecture","Self-Evolving AI Architecture","नियंत्रित architecture improvement","Versioning, rollback और approval gates के साथ सुरक्षित सुधार।","Planned"},
            {"scientific-discovery-intelligence","Scientific Discovery Intelligence","वैज्ञानिक खोज और evidence","Hypothesis, experiment design और evidence tracking।","Planned"},
            {"superhuman-mathematical-engine","Superhuman Mathematical Engine","गणित, proofs और validation","Symbolic/numeric engine और reproducible benchmarks।","Planned"},
            {"autonomous-software-engineering","Autonomous Software Engineering","Code, tests और review","Repository-aware coding, tests और approval-based changes।","Early tooling"},
            {"multi-agent-superintelligence","Multi-Agent Superintelligence","विशेषज्ञ agents का coordination","Task decomposition, communication और conflict resolution।","Planned"},
            {"advanced-memory-and-knowledge","Advanced Memory and Knowledge","Knowledge और secure memory","Provenance, retrieval, privacy और secure persistence।","Basic prototype"},
            {"continuous-learning-and-training","Continuous Learning and Training","Training pipelines और safe learning","Dataset validation, checkpoints और regression monitoring।","Toy training only"},
            {"beyond-gpt-6-benchmark-lab","Beyond GPT-6 Benchmark Lab","Fair benchmark और measurable evidence","Reasoning, coding, math, science और safety benchmarks।","Planned"},
            {"extreme-engineering-optimization","Extreme Engineering Optimization","Speed, memory और compute","Latency, throughput, memory और energy optimization।","Planned"},
            {"security-and-control","Security and Control","Sandbox, permissions और audit","Least privilege, logs, threat modeling और security tests।","Initial controls"},
            {"autonomous-project-execution","Autonomous Project Execution","Task tracking और bounded execution","Task planning, progress reports और approval checkpoints।","Initial automation"},
            {"scientific-proof","99% Ambition — Scientific Proof","हर दावे के लिए reproducible proof","99% लक्ष्य है; स्वतंत्र प्रमाण के बिना उपलब्धि claim नहीं होगी।","Validation pending"}
        };
        modules.clear();
        for(int i=0;i<rows.length;i++) {
            Module m=new Module(new JSONObject());
            m.id=rows[i][0];m.title=rows[i][1];m.summary=rows[i][2];m.details=rows[i][3];m.status=rows[i][4];
            m.anchor="#"+String.format(java.util.Locale.US,"%02d",i+1)+"-"+m.id;
            modules.add(m);
        }
    }

    private void syncDashboard() {
        syncStatus.setText("Syncing dashboard configuration…");
        new Thread(() -> {
            HttpURLConnection c=null;
            try {
                c=(HttpURLConnection)new URL(RAW_CONFIG+"?v="+System.currentTimeMillis()).openConnection();
                c.setConnectTimeout(9000);c.setReadTimeout(9000);c.setRequestProperty("Cache-Control","no-cache");
                if(c.getResponseCode()!=200) throw new Exception("HTTP "+c.getResponseCode());
                BufferedReader r=new BufferedReader(new InputStreamReader(c.getInputStream(),"UTF-8"));
                StringBuilder body=new StringBuilder();String line;
                while((line=r.readLine())!=null) body.append(line);
                JSONObject config=new JSONObject(body.toString());
                JSONArray arr=config.getJSONArray("modules");
                ArrayList<Module> fresh=new ArrayList<>();
                for(int i=0;i<arr.length();i++) fresh.add(new Module(arr.getJSONObject(i)));
                if(fresh.size()!=14) throw new Exception("Expected 14 modules");
                runOnUiThread(() -> {
                    modules.clear();modules.addAll(fresh);renderModules();
                    versionText.setText("Dashboard config v"+config.optInt("config_version",1)+" • synced from repository");
                    syncStatus.setText("LIVE CONFIG SYNCED • "+config.optString("updated_at",""));
                });
            } catch(Exception e) {
                runOnUiThread(() -> syncStatus.setText("Offline mode • showing saved 14-module dashboard"));
            } finally { if(c!=null)c.disconnect(); }
        }).start();
    }

    private void renderModules() {
        if(moduleList==null) return;
        moduleList.removeAllViews();
        for(int i=0;i<modules.size();i++) {
            final Module m=modules.get(i);
            LinearLayout card=panel(Color.WHITE,15);
            LinearLayout top=new LinearLayout(this);
            top.setOrientation(LinearLayout.HORIZONTAL);
            TextView number=label(String.format(java.util.Locale.US,"%02d",i+1),13,ACCENT,true);
            number.setGravity(Gravity.CENTER);
            GradientDrawable bubble=new GradientDrawable();bubble.setColor(Color.rgb(239,237,255));bubble.setCornerRadius(dp(12));
            number.setBackground(bubble);number.setPadding(dp(10),dp(8),dp(10),dp(8));
            top.addView(number,new LinearLayout.LayoutParams(dp(42),dp(40)));
            LinearLayout titleBox=new LinearLayout(this);titleBox.setOrientation(LinearLayout.VERTICAL);
            TextView name=label(m.title,15,INK,true);
            TextView summary=label(m.summary,11,MUTED,false);
            titleBox.addView(name);titleBox.addView(summary);
            LinearLayout.LayoutParams tlp=new LinearLayout.LayoutParams(0,-2,1);tlp.leftMargin=dp(11);
            top.addView(titleBox,tlp);
            card.addView(top);
            TextView state=label("STATUS  •  "+m.status.toUpperCase(java.util.Locale.ROOT),10,TEAL,true);
            state.setPadding(0,dp(10),0,dp(5));card.addView(state);
            Button open=actionButton("OPEN MODULE  →",Color.rgb(239,237,255),ACCENT);
            open.setOnClickListener(v -> showModule(m));
            card.addView(open);
            moduleList.addView(card,bottom(10));
        }
    }

    private void showModule(Module m) {
        String message=m.details+"\n\nCurrent status: "+m.status+"\n\nDashboard button opens this module's roadmap and code search.";
        new AlertDialog.Builder(this).setTitle(m.title).setMessage(message)
            .setPositiveButton("OPEN ROADMAP",(d,w)->open(REPO+"/blob/android-control-app-v1/docs/roadmap/14-point-roadmap.md"+m.anchor))
            .setNeutralButton("SEARCH MODULE CODE",(d,w)->open(REPO+"/search?q="+Uri.encode(m.id)+"&type=code"))
            .setNegativeButton("CLOSE",null).show();
    }

    private TextView label(String s,int size,int color,boolean bold) {
        TextView t=new TextView(this);t.setText(s);t.setTextSize(size);t.setTextColor(color);
        t.setPadding(0,dp(3),0,dp(3));if(bold)t.setTypeface(Typeface.DEFAULT,Typeface.BOLD);return t;
    }
    private LinearLayout panel(int color,int radius) {
        LinearLayout box=new LinearLayout(this);box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(15),dp(14),dp(15),dp(14));
        GradientDrawable bg=new GradientDrawable();bg.setColor(color);bg.setCornerRadius(dp(radius));
        bg.setStroke(dp(1),Color.rgb(231,235,243));box.setBackground(bg);return box;
    }
    private Button actionButton(String text,int background,int foreground) {
        Button b=new Button(this);b.setText(text);b.setTextSize(11);b.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        b.setTextColor(foreground);b.setAllCaps(false);b.setMinHeight(dp(46));b.setPadding(dp(8),0,dp(8),0);
        GradientDrawable bg=new GradientDrawable();bg.setColor(background);bg.setCornerRadius(dp(12));b.setBackground(bg);
        return b;
    }
    private void addWideAction(LinearLayout parent,String title,String url) {
        Button b=actionButton(title,Color.rgb(245,247,251),INK);b.setOnClickListener(v->open(url));
        LinearLayout.LayoutParams lp=bottom(6);parent.addView(b,lp);
    }
    private TextView space(int h) { TextView v=new TextView(this);v.setHeight(dp(h));return v; }
    private LinearLayout.LayoutParams bottom(int h) {
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2);lp.bottomMargin=dp(h);return lp;
    }
    private void open(String url) {
        try { startActivity(new Intent(Intent.ACTION_VIEW,Uri.parse(url))); }
        catch(Exception e) { Toast.makeText(this,"Unable to open link",Toast.LENGTH_SHORT).show(); }
    }
}
