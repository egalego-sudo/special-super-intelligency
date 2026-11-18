package com.bittupandit.ssi;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import android.content.Intent;
import android.net.Uri;
import org.json.JSONObject;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MainActivity extends Activity {
    private final String repoUrl = "https://github.com/egalego-sudo/special-super-intelligency";
    private TextView status, commit, refreshed;
    private LinearLayout root;
    private int pad(float v) { return (int)(v * getResources().getDisplayMetrics().density + 0.5f); }

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().setStatusBarColor(Color.rgb(10, 18, 38));
        getWindow().setNavigationBarColor(Color.rgb(10, 18, 38));
        getWindow().getDecorView().setSystemUiVisibility(0);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad(20), pad(28), pad(20), pad(20));
        root.setBackgroundColor(Color.rgb(10,18,38));
        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);

        TextView brand = text("SPECIAL SUPER INTELLIGENCY", 22, Color.WHITE, true);
        root.addView(brand);
        root.addView(text("CONTROL CENTER  •  ANDROID", 12, Color.rgb(117,190,255), true));
        root.addView(space(22));
        card("PROJECT FOUNDATION", "100%  •  20/20 initial foundation packages verified", "This is repository milestone progress, not completion of the full AI system.");
        card("CREATOR", "Creator by Bittu Pandit", "Independent AI research and engineering");
        status = text("Repository status: loading…", 15, Color.WHITE, true);
        commit = text("Latest commit: checking…", 13, Color.LTGRAY, false);
        refreshed = text("", 11, Color.GRAY, false);
        cardViews("LIVE REPOSITORY", status, commit, refreshed);
        root.addView(space(12));
        addButton("↻  REFRESH REPOSITORY STATUS", v -> refresh());
        addButton("OPEN PROJECT REPOSITORY", v -> open(repoUrl));
        addButton("OPEN CODE & UPDATE FILES", v -> open(repoUrl + "/tree/main"));
        addButton("OPEN TESTS & BUILD HISTORY", v -> open(repoUrl + "/actions"));
        addButton("OPEN PROJECT PROGRESS REPORT", v -> open(repoUrl + "/blob/main/PROGRESS.md"));
        root.addView(space(12));
        card("SECURITY", "Sandbox controls are part of the repository", "This app does not execute code on your phone or control a VM. Keep experiments in the isolated runner.");
        card("APP VERSION", "Version 1.0 • First control dashboard", "Repository updates require GitHub sign-in in the browser. No access token is stored inside this APK.");
        root.addView(text("Creator by Bittu Pandit", 12, Color.rgb(117,190,255), true));
        refresh();
    }

    private TextView text(String s, int size, int color, boolean bold) {
        TextView t = new TextView(this); t.setText(s); t.setTextSize(size); t.setTextColor(color);
        t.setPadding(0, pad(5), 0, pad(5));
        if (bold) t.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return t;
    }
    private View space(int h) { View v=new View(this); v.setLayoutParams(new LinearLayout.LayoutParams(1,pad(h))); return v; }
    private void card(String title, String line, String note) {
        TextView a=text(title,12,Color.rgb(117,190,255),true);
        TextView b=text(line,17,Color.WHITE,true);
        TextView c=text(note,12,Color.LTGRAY,false);
        LinearLayout box=new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(pad(16),pad(12),pad(16),pad(12)); box.setBackgroundColor(Color.rgb(23,35,62));
        box.addView(a); box.addView(b); box.addView(c);
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.bottomMargin=pad(12); root.addView(box,lp);
    }
    private void cardViews(String title, View... views) {
        LinearLayout box=new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(pad(16),pad(12),pad(16),pad(12)); box.setBackgroundColor(Color.rgb(23,35,62));
        box.addView(text(title,12,Color.rgb(117,190,255),true));
        for(View v:views) box.addView(v);
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.bottomMargin=pad(12); root.addView(box,lp);
    }
    private void addButton(String label, View.OnClickListener click) {
        Button b=new Button(this); b.setText(label); b.setTextColor(Color.WHITE);
        b.setBackgroundTintList(android.content.res.ColorStateList.valueOf(Color.rgb(35,91,160)));
        b.setOnClickListener(click);
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.bottomMargin=pad(8); root.addView(b,lp);
    }
    private void open(String url) { startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url))); }
    private void refresh() {
        status.setText("Repository status: checking GitHub…");
        new Thread(() -> {
            HttpURLConnection c=null;
            try {
                c=(HttpURLConnection)new URL("https://api.github.com/repos/egalego-sudo/special-super-intelligency").openConnection();
                c.setConnectTimeout(10000); c.setReadTimeout(10000); c.setRequestProperty("Accept","application/vnd.github+json");
                BufferedReader r=new BufferedReader(new InputStreamReader(c.getInputStream()));
                StringBuilder s=new StringBuilder(); String line; while((line=r.readLine())!=null)s.append(line);
                JSONObject j=new JSONObject(s.toString());
                String branch=j.optString("default_branch","main");
                HttpURLConnection cc=(HttpURLConnection)new URL("https://api.github.com/repos/egalego-sudo/special-super-intelligency/commits/"+branch).openConnection();
                cc.setConnectTimeout(10000); cc.setReadTimeout(10000);
                BufferedReader rr=new BufferedReader(new InputStreamReader(cc.getInputStream()));
                StringBuilder ss=new StringBuilder(); while((line=rr.readLine())!=null)ss.append(line);
                JSONObject cj=new JSONObject(ss.toString()); String sha=cj.optString("sha","unknown");
                String msg=cj.getJSONObject("commit").getJSONObject("message").optString("","Latest change");
                final String fbranch=branch, fsha=sha.substring(0,Math.min(7,sha.length())), fmsg=msg;
                runOnUiThread(() -> { status.setText("Repository: ONLINE  •  "+fbranch); commit.setText("Latest: "+fsha+"  •  "+fmsg); refreshed.setText("Status refreshed from GitHub"); });
            } catch(Exception e) { runOnUiThread(() -> { status.setText("Repository status: unavailable"); commit.setText("Check internet and tap Refresh."); refreshed.setText("GitHub public status could not be loaded."); }); }
            finally { if(c!=null)c.disconnect(); }
        }).start();
    }
}
